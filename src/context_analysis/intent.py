"""Intent analysis for conversation understanding

Intent is a question about meaning, so the primary path is semantic: LLM
classifiers, one per field, each with its own PromptManager-managed prompt
(and therefore versioned in Langfuse):

  - "intent_user_classification"            -> user_intents / primary_intent
    (or "intent_user_classification_closed_set" when a fixed taxonomy is set)
  - "intent_assistant_objectives"           -> assistant_objectives
  - "intent_alignment_scoring"              -> intent_alignment_score

Intent is read by the LLM classifier and by nothing else: there is no
deterministic intent classifier behind it. Earlier versions kept one -- first
a hand-written keyword list, then a matcher over the taxonomy's own labels --
and both were removed. Measured over the 27.7k-utterance Bitext goldenset by
`scripts/eval_intent_accuracy.py`, the keyword list scored 0.29 and the
matcher 0.85 against the semantic path's 0.94, and neither reads meaning:
they read spellings, so they answer confidently to paraphrase and to typos
alike. A second label space to keep in sync was not worth that.

So a failed user-intent call has no second opinion to fall back on, and says
so: `primary_intent` is "unknown", `user_intents` is empty, and the prompt's
name is listed in "classifier_failures". Alignment then has no intents to
score and reports itself undefined rather than zero. The other two fields do
still degrade deterministically -- `assistant_objectives` to a keyword map
and `intent_alignment_score` to `_OBJECTIVE_COMPATIBILITY` plus TF-IDF
similarity -- because those answer a question about an assistant turn that
is already in hand, not about what the user meant.

`taxonomy` sets the label space: open-vocabulary (no taxonomy) the
classifier emits the verbs in `INTENT_VOCABULARY`; given a closed set -- a
product's own intents, or a labeled goldenset's -- it is pinned to that set
by prompt and anything off-list is rejected.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.shared.llm.classifier import (
    JSONClassifier,
    clean_confidence,
    clean_score,
    clean_string_list,
)
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager
from src.shared.logger import get_logger

logger = get_logger(__name__)

_MAX_USER_INTENTS = 5
_MAX_ASSISTANT_OBJECTIVES = 5
_MAX_EVIDENCE_CHARS = 400

# Alignment is undefined, not zero, when there is nothing to compare: a
# single user utterance has no assistant turn to be aligned with.
_UNDEFINED_ALIGNMENT = 0.5

UNKNOWN_INTENT = "unknown"

# Preferred open-vocabulary labels, interpolated into the classifier prompt.
# Single lowercase verbs because `rule_generator/classifier.py` matches
# `user_intents` against its keyword map by exact equality -- a label like
# "explain_concept" or "Explain" would never match. The list deliberately
# includes the verbs that map hits ("explain", "clarify", "implement",
# "write", "create", "summarize", "follow", "answer"), and every one of them
# has an entry in `_OBJECTIVE_COMPATIBILITY`.
INTENT_VOCABULARY = (
    "help",
    "explain",
    "clarify",
    "answer",
    "fix",
    "troubleshoot",
    "implement",
    "write",
    "create",
    "summarize",
    "translate",
    "analyze",
    "review",
    "compare",
    "recommend",
    "decide",
    "optimize",
    "configure",
    "follow",
    "learn",
)

# Which assistant objectives count as engaging with which user intent. Used
# only by the fallback alignment path, in place of the old implementation's
# intersection of two vocabularies that could never intersect.
#
# Every set must therefore contain at least one label from
# `_FALLBACK_OBJECTIVE_KEYWORDS` (the only labels the fallback path can
# produce) alongside the richer verbs the LLM path emits -- otherwise this
# map reintroduces the very bug it replaces, for that intent. Near-synonyms
# across the two vocabularies ("validate"/"verify") are both listed.
# `tests/test_context.py::TestObjectiveCompatibility` holds this invariant.
_OBJECTIVE_COMPATIBILITY = {
    "help": {"assist", "implement", "suggest", "inform", "redirect"},
    "explain": {"inform", "explain", "clarify", "describe"},
    "clarify": {"inform", "explain", "clarify", "verify", "validate"},
    "answer": {"inform", "explain", "answer"},
    "fix": {"implement", "assist", "verify", "validate", "troubleshoot"},
    "troubleshoot": {
        "implement",
        "assist",
        "verify",
        "validate",
        "troubleshoot",
    },
    "implement": {"implement", "assist", "write"},
    "write": {"implement", "write", "assist"},
    "create": {"implement", "write", "assist"},
    "summarize": {"inform", "describe", "summarize"},
    "translate": {"implement", "write", "inform"},
    "analyze": {"inform", "verify", "validate", "describe"},
    "review": {"verify", "validate", "suggest", "inform"},
    "compare": {"inform", "describe", "suggest"},
    "recommend": {"suggest", "recommend", "inform"},
    "decide": {"suggest", "recommend", "inform"},
    "optimize": {"suggest", "implement", "recommend"},
    "configure": {"implement", "assist", "inform"},
    "follow": {"implement", "assist", "inform"},
    "learn": {"inform", "explain", "describe"},
}

# The objectives fallback. Substring matching, reached only when that
# classifier's call or output fails; there is no such map for intents.
_FALLBACK_OBJECTIVE_KEYWORDS = {
    "inform": ["explain", "inform", "provide", "describe", "tell"],
    "assist": ["help", "assist", "guide", "support", "enable"],
    "validate": ["validate", "check", "verify", "confirm"],
    "suggest": ["suggest", "recommend", "propose", "advise"],
    "implement": ["implement", "create", "build", "develop", "write"],
}

# Explicit speaker markers. Roles are read from these rather than from
# `shared.utils.parse_conversation_turns`, which defaults an unlabeled turn
# to "assistant" -- that would make a bare user utterance (one goldenset row,
# say) look like an assistant turn and invent objectives for it.
_ROLE_PATTERN = re.compile(r"^[ \t]*(user|assistant)[ \t]*:", re.IGNORECASE | re.MULTILINE)

_LABEL_SANITIZER = re.compile(r"[^a-z0-9_]+")


class IntentAnalyzer:
    """Analyzes user and assistant intents in conversation"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        taxonomy: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            llm_client: LLM client for the semantic classifiers
            prompt_manager: Prompt source for those classifiers
            taxonomy: Closed set of allowed intent labels. When given, the
                classifier must pick `primary_intent` from it and anything
                off-list is rejected -- used to score the analyzer against a
                labeled goldenset, or to pin it to a product's own intents.
        """
        self.taxonomy = [str(label).strip() for label in taxonomy] if taxonomy else None
        # Labels are compared after sanitizing, because that is the form
        # `_normalize_label` produces: a taxonomy spelled "Cancel Order"
        # could otherwise never be matched by its own classifier's output.
        self._allowed_labels = (
            {
                label
                for label in (_sanitize_token(raw) for raw in self.taxonomy)
                if label
            }
            if self.taxonomy
            else None
        )
        self.classifier = JSONClassifier(
            llm_client=llm_client, prompt_manager=prompt_manager
        )

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Analyze intents in conversation

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with intent analysis
        """
        failures: List[str] = []
        user_text, assistant_text = _speaker_texts(conversation)

        user_result = self._classify_user_intents(conversation, failures)

        # No assistant turn means there are no objectives to read and no
        # alignment to score, so both classifiers are skipped rather than
        # asked to invent an answer. This also keeps single-utterance
        # analysis down to one LLM call.
        if assistant_text:
            objectives, objectives_confidence = self._classify_objectives(
                conversation, assistant_text, failures
            )
            alignment = self._score_alignment(
                conversation,
                user_result["user_intents"],
                objectives,
                user_text,
                assistant_text,
                failures,
            )
        else:
            objectives, objectives_confidence = [], 0.0
            alignment = {
                "score": _UNDEFINED_ALIGNMENT,
                "rationale": "No assistant turn to align with.",
                "unmet_intents": [],
            }

        analysis = {
            "user_intents": user_result["user_intents"],
            "user_intent_details": user_result["details"],
            "assistant_objectives": objectives,
            "assistant_objectives_confidence": objectives_confidence,
            "intent_alignment_score": alignment["score"],
            "intent_alignment_rationale": alignment["rationale"],
            "unmet_intents": alignment["unmet_intents"],
            "intent_segments": user_result["segments"],
            "primary_intent": user_result["primary_intent"],
            "primary_intent_confidence": user_result["confidence"],
            "classifier_failures": failures,
        }

        logger.debug(f"Intent analysis complete: {analysis}")
        return analysis

    # ------------------------------------------------------------------
    # User intents
    # ------------------------------------------------------------------

    def _classify_user_intents(
        self, conversation: str, failures: List[str]
    ) -> Dict[str, Any]:
        """Read what the user wants

        This is the only path to an intent, so a failure here is reported as
        one -- "unknown" with an empty intent list -- rather than papered over
        with a guess from a second, weaker classifier.
        """
        if self.taxonomy:
            template = "intent_user_classification_closed_set"
            parsed = self.classifier.call(
                template,
                "context_analysis.intent_user_closed_set",
                failures,
                conversation=conversation,
                taxonomy="\n".join(f"- {label}" for label in self.taxonomy),
                max_intents=_MAX_USER_INTENTS,
            )
        else:
            template = "intent_user_classification"
            parsed = self.classifier.call(
                template,
                "context_analysis.intent_user",
                failures,
                conversation=conversation,
                vocabulary="\n".join(
                    f"- {label}" for label in INTENT_VOCABULARY
                ),
                max_intents=_MAX_USER_INTENTS,
            )

        if parsed is not None:
            result = self._read_user_intents(parsed)
            if result["user_intents"]:
                return result
            # A parseable response that named no usable intent -- every label
            # off-taxonomy, say -- is as much a failure as an unparseable one,
            # and `call` only records the failures it saw itself.
            logger.warning("User-intent classifier returned no usable label")
            failures.append(template)

        return {
            "user_intents": [],
            "details": [],
            "segments": {},
            "primary_intent": UNKNOWN_INTENT,
            "confidence": 0.0,
        }

    def _read_user_intents(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the user-intent classifier's output"""
        details: List[Dict[str, Any]] = []
        labels: List[str] = []
        segments: Dict[str, str] = {}

        raw_intents = parsed.get("user_intents")
        if isinstance(raw_intents, list):
            for entry in raw_intents:
                if isinstance(entry, dict):
                    label = self._normalize_label(entry.get("label"))
                    evidence = str(entry.get("evidence") or "").strip()
                    confidence = clean_confidence(entry.get("confidence"))
                else:
                    # Tolerate a bare list of labels: the field is still
                    # usable, just without evidence.
                    label = self._normalize_label(entry)
                    evidence, confidence = "", 0.0

                if not label or label in labels:
                    continue

                labels.append(label)
                details.append(
                    {
                        "label": label,
                        "evidence": evidence[:_MAX_EVIDENCE_CHARS],
                        "confidence": confidence,
                    }
                )
                if evidence:
                    segments[label] = evidence[:_MAX_EVIDENCE_CHARS]
                if len(labels) >= _MAX_USER_INTENTS:
                    break

        # `user_intents` and `user_intent_details` are positionally parallel,
        # so moving the primary intent to the front has to move its details
        # with it.
        primary = self._normalize_label(parsed.get("primary_intent"))
        if primary and primary not in labels:
            # The classifier named a primary intent it left out of the list;
            # keep it, and keep it first.
            labels.insert(0, primary)
            details.insert(
                0, {"label": primary, "evidence": "", "confidence": 0.0}
            )
        elif primary:
            index = labels.index(primary)
            labels.insert(0, labels.pop(index))
            details.insert(0, details.pop(index))

        return {
            "user_intents": labels,
            "details": details,
            "segments": segments,
            "primary_intent": labels[0] if labels else UNKNOWN_INTENT,
            "confidence": clean_confidence(parsed.get("confidence")),
        }

    def _normalize_label(self, value: Any) -> str:
        """Coerce a model-supplied label to a single comparable token

        Downstream consumers match these by exact equality, so the label is
        lowercased and stripped to `[a-z0-9_]`. In closed-set mode anything
        outside the taxonomy is dropped: an off-list label is a classifier
        failure, not a new intent.
        """
        if not isinstance(value, (str, int, float)):
            return ""

        label = _LABEL_SANITIZER.sub("_", str(value).strip().lower()).strip("_")
        if not label or label == UNKNOWN_INTENT:
            return ""

        if self._allowed_labels is not None and label not in self._allowed_labels:
            logger.warning(f"Dropping off-taxonomy intent label: {label!r}")
            return ""

        return label

    # ------------------------------------------------------------------
    # Assistant objectives
    # ------------------------------------------------------------------

    def _classify_objectives(
        self, conversation: str, assistant_text: str, failures: List[str]
    ) -> Tuple[List[str], float]:
        """Read what the assistant set out to do

        Unlike intents, this one does keep a keyword fallback: the question is
        what an assistant turn already in hand was doing, so matching its
        verbs is a weak answer but an honest one.
        """
        parsed = self.classifier.call(
            "intent_assistant_objectives",
            "context_analysis.intent_objectives",
            failures,
            conversation=conversation,
            max_objectives=_MAX_ASSISTANT_OBJECTIVES,
        )
        if parsed is not None:
            objectives = [
                label
                for label in (
                    _sanitize_token(value)
                    for value in clean_string_list(
                        parsed.get("assistant_objectives"),
                        limit=_MAX_ASSISTANT_OBJECTIVES,
                    )
                )
                if label
            ]
            if objectives:
                return objectives, clean_confidence(parsed.get("confidence"))

        return _match_keywords(
            assistant_text, _FALLBACK_OBJECTIVE_KEYWORDS
        ), 0.0

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _score_alignment(
        self,
        conversation: str,
        user_intents: List[str],
        objectives: List[str],
        user_text: str,
        assistant_text: str,
        failures: List[str],
    ) -> Dict[str, Any]:
        """Judge whether the assistant engaged with what the user wanted

        With no intents read -- the classifier failed -- there is nothing to
        align against, so the call is skipped and the fallback reports the
        score as undefined.
        """
        if user_intents:
            parsed = self.classifier.call(
                "intent_alignment_scoring",
                "context_analysis.intent_alignment",
                failures,
                conversation=conversation,
                user_intents=", ".join(user_intents),
                assistant_objectives=", ".join(objectives) or "(none stated)",
            )
            if parsed is not None and "alignment_score" in parsed:
                return {
                    "score": clean_score(
                        parsed.get("alignment_score"),
                        default=_UNDEFINED_ALIGNMENT,
                    ),
                    "rationale": str(parsed.get("rationale", "")).strip(),
                    "unmet_intents": [
                        label
                        for label in clean_string_list(
                            parsed.get("unmet_intents"),
                            limit=_MAX_USER_INTENTS,
                        )
                        if _sanitize_token(label) in user_intents
                    ],
                }

        return self._fallback_alignment(
            user_intents, objectives, user_text, assistant_text
        )

    def _fallback_alignment(
        self,
        user_intents: List[str],
        objectives: List[str],
        user_text: str,
        assistant_text: str,
    ) -> Dict[str, Any]:
        """Score alignment without an LLM, from two measurable signals

        Half the score is topical: the TF-IDF cosine similarity between what
        the user said and what the assistant said, which at least detects an
        assistant answering something else entirely. The other half is the
        share of user intents whose objective appears in
        `_OBJECTIVE_COMPATIBILITY` -- the replacement for the old intersection
        of two disjoint vocabularies.

        Intents from a closed-set taxonomy have no entry in that map and
        nothing in it stands in for them, so they are left out of the
        coverage half rather than counted unmet: a label this path cannot
        reason about must not produce a confident 0.0.
        """
        if not user_intents or not objectives:
            return {
                "score": _UNDEFINED_ALIGNMENT,
                "rationale": (
                    "Not enough signal to score alignment: "
                    f"{len(user_intents)} user intent(s), "
                    f"{len(objectives)} assistant objective(s)."
                ),
                "unmet_intents": list(user_intents),
            }

        objective_set = set(objectives)
        similarity = _topical_similarity(user_text, assistant_text)

        scorable = [
            intent for intent in user_intents if intent in _OBJECTIVE_COMPATIBILITY
        ]
        if not scorable:
            return {
                "score": round(min(max(similarity, 0.0), 1.0), 4),
                "rationale": (
                    f"Fallback scoring: none of the {len(user_intents)} user "
                    "intent(s) are in the objective-compatibility map "
                    "(closed-set labels), so the score is topical similarity "
                    f"{similarity:.2f} alone."
                ),
                "unmet_intents": [],
            }

        unmet = [
            intent
            for intent in scorable
            if not (_OBJECTIVE_COMPATIBILITY[intent] & objective_set)
        ]
        covered = 1.0 - (len(unmet) / len(scorable))
        score = 0.5 * covered + 0.5 * similarity

        return {
            "score": round(min(max(score, 0.0), 1.0), 4),
            "rationale": (
                f"Fallback scoring: {len(scorable) - len(unmet)}/"
                f"{len(scorable)} scorable intent(s) matched by a compatible "
                f"objective, topical similarity {similarity:.2f}."
            ),
            "unmet_intents": unmet,
        }


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def _sanitize_token(value: Any) -> str:
    """Lowercase a label and strip it to `[a-z0-9_]`"""
    if not isinstance(value, (str, int, float)):
        return ""
    return _LABEL_SANITIZER.sub("_", str(value).strip().lower()).strip("_")


def _match_keywords(text: str, keyword_map: Dict[str, List[str]]) -> List[str]:
    """Labels whose keywords appear in `text` (the objectives fallback)"""
    text_lower = (text or "").lower()
    return [
        label
        for label, keywords in keyword_map.items()
        if any(keyword in text_lower for keyword in keywords)
    ]


def _speaker_texts(conversation: str) -> Tuple[str, str]:
    """Split a conversation into (user text, assistant text)

    Roles come from explicit "User:"/"Assistant:" markers. Text with no
    marker at all is treated as the user's, because a bare utterance is
    something the user said.
    """
    conversation = conversation or ""
    matches = list(_ROLE_PATTERN.finditer(conversation))
    if not matches:
        return conversation.strip(), ""

    chunks: Dict[str, List[str]] = {"user": [], "assistant": []}

    preamble = conversation[: matches[0].start()].strip()
    if preamble:
        chunks["user"].append(preamble)

    for index, match in enumerate(matches):
        role = match.group(1).lower()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(conversation)
        )
        content = conversation[match.end() : end].strip()
        if content:
            chunks[role].append(content)

    return "\n\n".join(chunks["user"]), "\n\n".join(chunks["assistant"])


def _topical_similarity(user_text: str, assistant_text: str) -> float:
    """TF-IDF cosine similarity between the two sides of the conversation"""
    if not user_text.strip() or not assistant_text.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([user_text, assistant_text])
    except ValueError as e:
        # Raised when every token was a stopword, so there is no vocabulary
        # left to compare.
        logger.debug(f"No TF-IDF vocabulary for alignment similarity: {e}")
        return 0.0

    similarity = float(cosine_similarity(matrix[0], matrix[1])[0][0])
    return min(max(similarity, 0.0), 1.0)
