"""Measure IntentAnalyzer accuracy against the Bitext customer-service goldensets

The two Bitext CSVs in data/raw label each utterance with one of 27 domain
intents ("cancel_order", "get_refund", ...). `IntentAnalyzer` is constructed
in closed-set mode (`taxonomy=`) over exactly those intents, so this drives
the real component -- its prompt, its normalization, its off-taxonomy
rejection -- rather than a reimplementation of it, and no gold label is
rewritten anywhere in this script: what is reported is accuracy on the
goldenset's own 27 intents.

Two numbers: `accuracy` (`primary_intent` is the gold intent) and
`any_intent_accuracy` (the gold intent appears anywhere in the multi-label
`user_intents` list). The closed-set prompt asks for a single label, so the
two usually coincide; a gap means the classifier is hedging.

Every call goes to the LLM -- the analyzer has no deterministic intent path
to compare against anymore -- so use `--limit` and `--workers`.

Usage:
    python scripts/eval_intent_accuracy.py --limit 300
    python scripts/eval_intent_accuracy.py --limit 1080 --workers 12
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    CLASSIFIER_LLM_MAX_TOKENS,
    OUTPUT_DATA_DIR,
    RAW_DATA_DIR,
)
from src.context_analysis.intent import IntentAnalyzer  # noqa: E402
from src.shared.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_GOLDENSETS = [
    RAW_DATA_DIR
    / "20000-Utterances-Training-dataset-for-chatbots-virtual-assistant-Bitext-sample.csv",
    RAW_DATA_DIR / "Bitext_Sample_Customer_Service_Training_Dataset.csv",
]
DEFAULT_OUTPUT = OUTPUT_DATA_DIR / "intent_accuracy.json"

UNKNOWN = "unknown"

# The two files disagree on the spelling of two labels for the same intent
# (and on their category names). Canonicalized so an utterance is not counted
# as a different class depending on which file it came from.
INTENT_ALIASES = {"check_invoices": "check_invoice"}
CATEGORY_ALIASES = {
    "INVOICES": "INVOICE",
    "REFUNDS": "REFUND",
    "SHIPPING": "SHIPPING_ADDRESS",
}

# Bitext's per-utterance language-generation tags. Accuracy is reported per
# tag so a weak slice (typos, colloquial phrasing, negation) is visible
# instead of being averaged away.
FLAG_MEANINGS = {
    "B": "basic syntactic structure",
    "C": "coordinated clauses",
    "D": "indirect/embedded question",
    "E": "expanded abbreviations",
    "I": "interrogative",
    "K": "keyword mode (no sentence)",
    "L": "lexical/semantic variation",
    "M": "morphological variation",
    "P": "politeness markers",
    "Q": "colloquial phrasing",
    "S": "offensive/strong language",
    "V": "non-verbal",
    "W": "offensive language",
    "Z": "noise (typos, spelling errors)",
}


@dataclass
class GoldItem:
    """One labeled utterance from a goldenset file"""

    utterance: str
    intent: str
    category: str
    flags: str
    source: str


@dataclass
class Prediction:
    """One predictor's output for one gold item"""

    predicted: str
    candidates: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------
# Goldenset loading
# --------------------------------------------------------------------------


def load_goldenset(paths: Sequence[Path]) -> Tuple[List[GoldItem], Dict[str, Any]]:
    """Load, canonicalize and de-duplicate the Bitext CSVs

    Returns the items plus provenance stats (per-file counts, duplicates
    dropped, utterances whose label still conflicts after canonicalization).
    """
    by_utterance: Dict[str, GoldItem] = {}
    per_file: Dict[str, int] = {}
    duplicates = 0
    conflicts: List[Dict[str, str]] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Goldenset not found: {path}")

        # utf-8-sig: both files start with a BOM, which would otherwise end up
        # inside the first column name.
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

        per_file[path.name] = len(rows)

        for row in rows:
            utterance = (row.get("utterance") or "").strip()
            intent = (row.get("intent") or "").strip()
            if not utterance or not intent:
                continue

            intent = INTENT_ALIASES.get(intent, intent)
            category = (row.get("category") or "").strip().upper()
            category = CATEGORY_ALIASES.get(category, category)

            item = GoldItem(
                utterance=utterance,
                intent=intent,
                category=category,
                flags=(row.get("flags") or "").strip().upper(),
                source=path.name,
            )

            key = utterance.lower()
            existing = by_utterance.get(key)
            if existing is None:
                by_utterance[key] = item
                continue

            duplicates += 1
            if existing.intent != item.intent:
                conflicts.append(
                    {
                        "utterance": utterance,
                        "kept": existing.intent,
                        "dropped": item.intent,
                    }
                )

    items = list(by_utterance.values())
    stats = {
        "files": per_file,
        "rows_read": sum(per_file.values()),
        "unique_utterances": len(items),
        "duplicates_dropped": duplicates,
        "label_conflicts": conflicts,
        "intents": len({item.intent for item in items}),
    }
    return items, stats


def stratified_sample(
    items: List[GoldItem], limit: Optional[int], seed: int
) -> List[GoldItem]:
    """Sample `limit` items keeping every intent represented

    Round-robins across intents so a small sample does not drop rare intents
    (SHIPPING-type intents are ~1% of the corpus) and does not inherit the
    corpus's class imbalance wholesale.
    """
    if limit is None or limit >= len(items):
        return items

    rng = random.Random(seed)
    buckets: Dict[str, List[GoldItem]] = defaultdict(list)
    for item in items:
        buckets[item.intent].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    sampled: List[GoldItem] = []
    intents = sorted(buckets)
    while len(sampled) < limit:
        drained = True
        for intent in intents:
            if not buckets[intent]:
                continue
            drained = False
            sampled.append(buckets[intent].pop())
            if len(sampled) >= limit:
                break
        if drained:
            break

    rng.shuffle(sampled)
    return sampled


# --------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------


class IntentPredictor:
    """`IntentAnalyzer` pinned to the goldenset's taxonomy"""

    def __init__(self, taxonomy: Sequence[str], model: Optional[str] = None) -> None:
        from src.shared.llm.client import LLMClient

        # `fallbacks=()`: this report names one model, so a rate limit must
        # show up as a failed utterance in the accuracy number -- not as a
        # different vendor's model quietly answering part of the run.
        client = LLMClient(
            max_tokens=CLASSIFIER_LLM_MAX_TOKENS,
            fallbacks=(),
            **({"model": model} if model else {}),
        )
        # One analyzer (so one client, one prompt fetch) shared by the pool:
        # the analyzer holds no per-call state.
        self.analyzer = IntentAnalyzer(llm_client=client, taxonomy=sorted(taxonomy))
        self.client = client

    def predict(self, items: List[GoldItem], workers: int) -> List[Prediction]:
        if workers <= 1:
            return [self._predict_one(item) for item in items]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._predict_one, items))

    def _predict_one(self, item: GoldItem) -> Prediction:
        try:
            analysis = self.analyzer.analyze(item.utterance)
        except Exception as e:  # noqa: BLE001 - one bad call must not end the run
            logger.warning(f"Analysis failed for {item.utterance!r}: {e}")
            return Prediction(predicted=UNKNOWN, error=str(e))

        # The analyzer records its own degradations instead of raising; an
        # utterance it could not classify is a real failure mode and belongs
        # in the accuracy number, not in a traceback.
        failures = analysis.get("classifier_failures") or []
        return Prediction(
            predicted=analysis["primary_intent"],
            candidates=list(analysis["user_intents"]),
            confidence=analysis.get("primary_intent_confidence"),
            error=", ".join(failures) if failures else None,
        )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def per_label_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Precision, recall, F1 and support for every label seen in either list"""
    labels = sorted(set(gold) | set(predicted))
    true_positives: Counter = Counter()
    predicted_counts: Counter = Counter()
    gold_counts: Counter = Counter()

    for g, p in zip(gold, predicted):
        gold_counts[g] += 1
        predicted_counts[p] += 1
        if g == p:
            true_positives[g] += 1

    metrics = {}
    for label in labels:
        tp = true_positives[label]
        precision = tp / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = tp / gold_counts[label] if gold_counts[label] else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": gold_counts[label],
            "predicted": predicted_counts[label],
        }
    return metrics


def sliced_accuracy(
    gold: Sequence[str], predicted: Sequence[str], keys: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Accuracy within each slice named by `keys` (one key per item)"""
    totals: Counter = Counter()
    correct: Counter = Counter()
    for g, p, key in zip(gold, predicted, keys):
        totals[key] += 1
        if g == p:
            correct[key] += 1

    return {
        key: {
            "accuracy": round(correct[key] / totals[key], 4),
            "support": totals[key],
        }
        for key in sorted(totals, key=lambda k: -totals[k])
    }


def flag_accuracy(
    gold: Sequence[str], predicted: Sequence[str], items: Sequence[GoldItem]
) -> Dict[str, Dict[str, Any]]:
    """Accuracy per Bitext language tag (an utterance carries several)"""
    totals: Counter = Counter()
    correct: Counter = Counter()
    for g, p, item in zip(gold, predicted, items):
        for flag in set(item.flags):
            totals[flag] += 1
            if g == p:
                correct[flag] += 1

    return {
        flag: {
            "meaning": FLAG_MEANINGS.get(flag, "unknown tag"),
            "accuracy": round(correct[flag] / totals[flag], 4),
            "support": totals[flag],
        }
        for flag in sorted(totals, key=lambda f: -totals[f])
    }


def top_confusions(
    gold: Sequence[str], predicted: Sequence[str], top_n: int = 15
) -> List[Dict[str, Any]]:
    """The most frequent (gold, predicted) mismatches"""
    pairs: Counter = Counter(
        (g, p) for g, p in zip(gold, predicted) if g != p
    )
    return [
        {"gold": g, "predicted": p, "count": count}
        for (g, p), count in pairs.most_common(top_n)
    ]


def evaluate(
    items: List[GoldItem],
    predictions: List[Prediction],
    errors_to_keep: int,
) -> Dict[str, Any]:
    """Turn raw predictions into the full metrics report"""
    gold = [item.intent for item in items]
    predicted = [prediction.predicted for prediction in predictions]

    correct = sum(1 for g, p in zip(gold, predicted) if g == p)
    total = len(items)

    label_metrics = per_label_metrics(gold, predicted)
    gold_labels = [label for label in label_metrics if label_metrics[label]["support"]]
    macro_f1 = (
        sum(label_metrics[label]["f1"] for label in gold_labels) / len(gold_labels)
        if gold_labels
        else 0.0
    )
    macro_recall = (
        sum(label_metrics[label]["recall"] for label in gold_labels)
        / len(gold_labels)
        if gold_labels
        else 0.0
    )
    weighted_f1 = (
        sum(
            label_metrics[label]["f1"] * label_metrics[label]["support"]
            for label in gold_labels
        )
        / total
        if total
        else 0.0
    )

    report: Dict[str, Any] = {
        "evaluated": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "balanced_accuracy": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "unknown_predictions": sum(1 for p in predicted if p == UNKNOWN),
        "prediction_errors": sum(1 for p in predictions if p.error),
        "per_label": label_metrics,
        "by_gold_intent": sliced_accuracy(
            gold, predicted, [item.intent for item in items]
        ),
        "by_category": sliced_accuracy(
            gold, predicted, [item.category for item in items]
        ),
        "by_flag": flag_accuracy(gold, predicted, items),
        "top_confusions": top_confusions(gold, predicted),
    }

    # `user_intents` is a multi-label list; crediting only its first entry
    # understates what the classifier detected, so score the set too.
    if any(prediction.candidates for prediction in predictions):
        any_correct = sum(
            1
            for g, prediction in zip(gold, predictions)
            if g in prediction.candidates
        )
        total_candidates = sum(len(p.candidates) for p in predictions)
        report["any_intent_accuracy"] = (
            round(any_correct / total, 4) if total else 0.0
        )
        report["mean_candidates"] = (
            round(total_candidates / total, 2) if total else 0.0
        )

    confidences = [p.confidence for p in predictions if p.confidence is not None]
    if confidences:
        report["mean_confidence"] = round(sum(confidences) / len(confidences), 4)

    report["errors"] = [
        {"utterance": item.utterance, "error": prediction.error}
        for item, prediction in zip(items, predictions)
        if prediction.error
    ][:errors_to_keep]

    report["misclassified_examples"] = [
        {
            "utterance": item.utterance,
            "gold": g,
            "predicted": p,
            "gold_intent": item.intent,
        }
        for item, g, p in zip(items, gold, predicted)
        if g != p
    ][:errors_to_keep]

    return report


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_report(report: Dict[str, Any], goldenset_stats: Dict[str, Any]) -> None:
    """Print the headline numbers and the worst slices"""
    print()
    print("=" * 72)
    print(f"Intent accuracy -- IntentAnalyzer, closed set over "
          f"{goldenset_stats['intents']} gold intents "
          f"(model: {report.get('model') or 'default'})")
    print("=" * 72)
    print(
        f"Goldenset: {goldenset_stats['rows_read']} rows -> "
        f"{goldenset_stats['unique_utterances']} unique utterances, "
        f"{goldenset_stats['intents']} intents "
        f"({goldenset_stats['duplicates_dropped']} duplicates dropped)"
    )
    print(f"Evaluated: {report['evaluated']} utterances")
    print()
    print(f"  accuracy           {report['accuracy']:.4f} "
          f"({report['correct']}/{report['evaluated']})")
    if "any_intent_accuracy" in report:
        print(f"  any-label accuracy {report['any_intent_accuracy']:.4f} "
              f"(gold label anywhere in user_intents; "
              f"mean {report['mean_candidates']} labels/utterance)")
    print(f"  balanced accuracy  {report['balanced_accuracy']:.4f} (macro recall)")
    print(f"  macro F1           {report['macro_f1']:.4f}")
    print(f"  weighted F1        {report['weighted_f1']:.4f}")
    if report["unknown_predictions"]:
        print(f"  unknown/failed     {report['unknown_predictions']} "
              f"({report['prediction_errors']} call or parse errors)")
    if "mean_confidence" in report:
        print(f"  mean confidence    {report['mean_confidence']:.4f}")

    print()
    print("Per-label (gold labels only, worst F1 first):")
    print(f"  {'label':<28} {'P':>7} {'R':>7} {'F1':>7} {'n':>6}")
    scored = [
        (label, metrics)
        for label, metrics in report["per_label"].items()
        if metrics["support"]
    ]
    for label, metrics in sorted(scored, key=lambda kv: kv[1]["f1"]):
        print(
            f"  {label:<28} {metrics['precision']:>7.3f} "
            f"{metrics['recall']:>7.3f} {metrics['f1']:>7.3f} "
            f"{metrics['support']:>6}"
        )

    print()
    print("Accuracy by Bitext language tag (lowest first):")
    for flag, slice_metrics in sorted(
        report["by_flag"].items(), key=lambda kv: kv[1]["accuracy"]
    ):
        print(
            f"  {flag} {slice_metrics['meaning']:<34} "
            f"{slice_metrics['accuracy']:>7.3f}  n={slice_metrics['support']}"
        )

    if report["top_confusions"]:
        print()
        print("Top confusions (gold -> predicted):")
        for confusion in report["top_confusions"]:
            print(
                f"  {confusion['gold']:<24} -> "
                f"{confusion['predicted']:<24} {confusion['count']:>5}"
            )
    print()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure IntentAnalyzer accuracy against the Bitext "
            "customer-service goldensets"
        )
    )
    parser.add_argument(
        "--goldenset",
        type=Path,
        nargs="+",
        default=DEFAULT_GOLDENSETS,
        help="Goldenset CSV files (default: both Bitext samples in data/raw)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate a stratified sample of this many utterances "
            "(default: all ~27.7k, which is one LLM call each)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Sampling seed (default: 42)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel LLM calls (default: 4)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model the analyzer's classifier runs on",
    )
    parser.add_argument(
        "--errors",
        type=int,
        default=30,
        help="Misclassified examples to keep in the JSON report (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON report (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    items, goldenset_stats = load_goldenset(args.goldenset)
    logger.info(
        f"Loaded {len(items)} unique utterances from "
        f"{len(goldenset_stats['files'])} goldenset file(s)"
    )
    if goldenset_stats["label_conflicts"]:
        logger.warning(
            f"{len(goldenset_stats['label_conflicts'])} utterance(s) carry "
            "different labels across files; kept the first occurrence"
        )

    evaluated_items = stratified_sample(items, args.limit, args.seed)

    # The taxonomy comes from the whole goldenset, not the sample, so a small
    # --limit does not quietly make the task easier.
    taxonomy = sorted({item.intent for item in items})

    predictor = IntentPredictor(taxonomy, model=args.model)

    logger.info(
        f"Classifying {len(evaluated_items)} utterances over "
        f"{len(taxonomy)} intents with {args.workers} worker(s)"
    )
    started = time.time()
    predictions = predictor.predict(evaluated_items, args.workers)
    elapsed = time.time() - started

    report = evaluate(evaluated_items, predictions, args.errors)
    report["goldenset"] = goldenset_stats
    report["elapsed_seconds"] = round(elapsed, 2)
    report["sampling"] = {
        "limit": args.limit,
        "seed": args.seed,
        "stratified": args.limit is not None and args.limit < len(items),
    }
    report["model"] = getattr(predictor.client, "model", args.model)
    report["llm_stats"] = {
        key: value
        for key, value in predictor.client.get_stats().items()
        if key != "last_call_time"
    }

    print_report(report, goldenset_stats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report -> {args.output}")


if __name__ == "__main__":
    main()
