"""Run IntentAnalyzer over a dataset of Langfuse-style conversation items"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_DIR, OUTPUT_DATA_DIR  # noqa: E402
from src.context_analysis.intent import IntentAnalyzer  # noqa: E402
from src.shared.logger import get_logger  # noqa: E402
from src.shared.utils import build_conversation_text  # noqa: E402

logger = get_logger(__name__)

DEFAULT_DATASET = RAW_DATA_DIR / "diana-agent-csat.json"
DEFAULT_OUTPUT = OUTPUT_DATA_DIR / "diana-agent-csat_intent_analysis.json"


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load a dataset export (list of items with input.query.content messages)"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Run intent analysis on every item in the dataset"""
    dataset = load_dataset(dataset_path)
    analyzer = IntentAnalyzer()

    results = []
    for item in dataset:
        conversation = build_conversation_text(item)
        analysis = analyzer.analyze(conversation)
        results.append(
            {
                "id": item.get("id"),
                "primary_intent": analysis["primary_intent"],
                "user_intents": analysis["user_intents"],
                "assistant_objectives": analysis["assistant_objectives"],
                "intent_alignment_score": analysis["intent_alignment_score"],
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run context_analysis/intent.py's IntentAnalyzer over a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to dataset JSON file (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write results JSON file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    logger.info(f"Running intent analysis on {args.dataset}")
    results = analyze_dataset(args.dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote {len(results)} results to {args.output}")
    print(f"Analyzed {len(results)} items -> {args.output}")


if __name__ == "__main__":
    main()
