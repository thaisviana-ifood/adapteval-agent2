"""Semantic analysis of conversation content

Two kinds of signal, each handled by the tool that does it better:

- **Statistical** (no LLM call): `topics` is ranked by TF-IDF over the
  conversation's turns, the same way `structural.py` ranks its per-speaker
  terms. Token counts, vocabulary diversity and the emoji tally are plain
  arithmetic.
- **LLM classifiers**, one per field that needs actual understanding, each
  with its own prompt managed through PromptManager (and therefore versioned
  in Langfuse):

  - "semantic_key_phrase_extraction"    -> key_phrases
  - "semantic_summary_generation"       -> summary
  - "semantic_sentiment_classification" -> sentiment_score / label / rationale

Each classifier degrades independently: if its call or its output fails to
parse, that field comes back empty/neutral, the prompt's name is listed in
"classifier_failures", and the other fields are unaffected. The statistical
fields never degrade.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import CLASSIFIER_LLM_MAX_TOKENS
from src.shared.emoji import analyze_emoji_sentiment
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager
from src.shared.logger import get_logger
from src.shared.utils import parse_conversation_turns, preprocess_text

logger = get_logger(__name__)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

_MAX_TOPICS = 5
_MAX_KEY_PHRASES = 10
_MAX_SUMMARY_SENTENCES = 3
_NEUTRAL_SENTIMENT = 0.5
_SENTIMENT_LABELS = {"positive", "neutral", "negative", "mixed"}

# Alphabetic tokens of 3+ characters: drops numbers, punctuation and
# 1-2 character noise, but keeps short terms that downstream task
# classification matches on exactly (e.g. "code", "step") -- the old
# wordlist-based filter required 5+ characters and silently dropped those.
_TOPIC_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"


class SemanticAnalyzer:
    """Analyzes semantic content, topics and tone"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        self.llm_client = llm_client or LLMClient(
            max_tokens=CLASSIFIER_LLM_MAX_TOKENS
        )
        self.prompt_manager = prompt_manager or PromptManager()

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Analyze semantic patterns in conversation

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with semantic metrics
        """
        failures: List[str] = []

        # Statistical: no LLM call
        topics = self._extract_topics(conversation)

        # One LLM classifier per field that needs understanding
        key_phrases, key_phrases_confidence = self._extract_key_phrases(
            conversation, failures
        )
        summary, summary_confidence = self._generate_summary(
            conversation, failures
        )
        sentiment = self._classify_sentiment(conversation, failures)

        # Vocabulary diversity
        vocab_diversity = self._calculate_vocabulary_diversity(
            conversation
        )

        # Emoji sentiment (positive/negative/neutral)
        emoji_sentiment = analyze_emoji_sentiment(conversation)

        analysis = {
            "key_phrases": key_phrases,
            "key_phrases_confidence": key_phrases_confidence,
            "summary": summary,
            "summary_confidence": summary_confidence,
            "sentiment_score": sentiment["sentiment_score"],
            "sentiment_label": sentiment["label"],
            "sentiment_confidence": sentiment["confidence"],
            "sentiment_rationale": sentiment["rationale"],
            "topics": topics,
            "vocabulary_diversity": vocab_diversity,
            "total_tokens": len(conversation.split()),
            "unique_tokens": len(set(conversation.lower().split())),
            "emoji_sentiment": emoji_sentiment,
            "classifier_failures": failures,
        }

        logger.debug(f"Semantic analysis complete: {analysis}")
        return analysis

    def _extract_topics(self, conversation: str) -> List[str]:
        """Rank the conversation's topics by mean TF-IDF across its turns

        Single words rather than phrases, because `rule_generator/classifier.py`
        matches `topics` against its keyword map by exact equality -- a bigram
        like "machine learning" would never match. English stopwords carry the
        filtering job the bundled wordlist used to do, so topics no longer
        depend on a 2 MB resource file.
        """
        documents = self._topic_documents(conversation)
        if not documents:
            return []

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                token_pattern=_TOPIC_TOKEN_PATTERN,
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
        except ValueError as e:
            # Raised when every token was a stopword, so the vocabulary is empty
            logger.debug(f"No TF-IDF vocabulary for topic extraction: {e}")
            return []

        feature_names = vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1

        ranked = sorted(
            zip(feature_names, mean_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [term for term, _ in ranked[:_MAX_TOPICS]]

    def _topic_documents(self, conversation: str) -> List[str]:
        """Split the conversation into the documents TF-IDF ranks over

        One document per turn, so a term repeated in every turn is discounted
        by IDF. A conversation with no turn markers stays a single document,
        where TF-IDF degenerates to term frequency.
        """
        turns = parse_conversation_turns(conversation)
        documents = [
            preprocess_text(turn["content"])
            for turn in turns
            if turn["content"].strip()
        ]
        if documents:
            return documents

        cleaned = preprocess_text(conversation)
        return [cleaned] if cleaned else []

    def _call_classifier(
        self,
        template_name: str,
        trace_name: str,
        failures: List[str],
        **variables: Any,
    ) -> Optional[Dict[str, Any]]:
        """Run one classifier prompt and return its parsed JSON object

        Returns None (and records the failure) if the prompt is missing, the
        call fails, or the response isn't a parseable JSON object.
        """
        prompt = self.prompt_manager.format_prompt(template_name, **variables)
        if not prompt:
            logger.warning(f"Prompt '{template_name}' unavailable")
            failures.append(template_name)
            return None

        try:
            response_text = self.llm_client.call(
                prompt,
                name=trace_name,
                langfuse_prompt=self.prompt_manager.get_prompt_client(
                    template_name
                ),
            )
        except Exception as e:
            logger.warning(f"Classifier '{template_name}' call failed: {e}")
            failures.append(template_name)
            return None

        parsed = self._parse_json_object(response_text, template_name)
        if parsed is None:
            failures.append(template_name)
        return parsed

    def _parse_json_object(
        self, response_text: Optional[str], template_name: str
    ) -> Optional[Dict[str, Any]]:
        """Extract the JSON object from a classifier response"""
        if not response_text:
            logger.warning(f"Classifier '{template_name}' returned nothing")
            return None

        match = _JSON_OBJECT_PATTERN.search(response_text)
        if not match:
            logger.warning(
                f"Classifier '{template_name}' returned no JSON object"
            )
            return None

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            logger.warning(
                f"Could not parse '{template_name}' JSON: {e}"
            )
            return None

        if not isinstance(parsed, dict):
            logger.warning(
                f"Classifier '{template_name}' returned {type(parsed).__name__}, "
                "expected an object"
            )
            return None

        return parsed

    def _extract_key_phrases(
        self, conversation: str, failures: List[str]
    ) -> Tuple[List[str], float]:
        """Extract the phrases carrying the conversation's substance"""
        parsed = self._call_classifier(
            "semantic_key_phrase_extraction",
            "context_analysis.semantic_key_phrases",
            failures,
            conversation=conversation,
            max_key_phrases=_MAX_KEY_PHRASES,
        )
        if parsed is None:
            return [], 0.0

        key_phrases = self._clean_string_list(
            parsed.get("key_phrases"), limit=_MAX_KEY_PHRASES
        )
        return key_phrases, self._clean_confidence(parsed.get("confidence"))

    def _generate_summary(
        self, conversation: str, failures: List[str]
    ) -> Tuple[str, float]:
        """Summarize what the user wanted and where the conversation ended up"""
        parsed = self._call_classifier(
            "semantic_summary_generation",
            "context_analysis.semantic_summary",
            failures,
            conversation=conversation,
            max_summary_sentences=_MAX_SUMMARY_SENTENCES,
        )
        if parsed is None:
            return "", 0.0

        summary = str(parsed.get("summary", "")).strip()
        return summary, self._clean_confidence(parsed.get("confidence"))

    def _classify_sentiment(
        self, conversation: str, failures: List[str]
    ) -> Dict[str, Any]:
        """Classify the user's overall sentiment across the conversation"""
        neutral = {
            "label": "neutral",
            "sentiment_score": _NEUTRAL_SENTIMENT,
            "confidence": 0.0,
            "rationale": "",
        }

        parsed = self._call_classifier(
            "semantic_sentiment_classification",
            "context_analysis.semantic_sentiment",
            failures,
            conversation=conversation,
        )
        if parsed is None:
            return neutral

        label = str(parsed.get("label", "")).strip().lower()
        if label not in _SENTIMENT_LABELS:
            logger.warning(f"Unexpected sentiment label: {label!r}")
            label = "neutral"

        return {
            "label": label,
            "sentiment_score": self._clean_score(
                parsed.get("sentiment_score"), default=_NEUTRAL_SENTIMENT
            ),
            "confidence": self._clean_confidence(parsed.get("confidence")),
            "rationale": str(parsed.get("rationale", "")).strip(),
        }

    def _clean_string_list(self, value: Any, limit: int) -> List[str]:
        """Normalize a classifier's string list: de-duplicated, capped, no blanks"""
        if not isinstance(value, list):
            return []

        cleaned: List[str] = []
        seen = set()
        for item in value:
            if not isinstance(item, (str, int, float)):
                continue
            text = str(item).strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            cleaned.append(text)
            if len(cleaned) >= limit:
                break

        return cleaned

    def _clean_score(self, value: Any, default: float) -> float:
        """Coerce a classifier's 0-1 score, clamping out-of-range values"""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return default
        return min(max(score, 0.0), 1.0)

    def _clean_confidence(self, value: Any) -> float:
        """Coerce a classifier's confidence; absent confidence means 0.0"""
        return self._clean_score(value, default=0.0)

    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (0-1)"""
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = len(set(words))
        return min(unique_words / len(words), 1.0)
