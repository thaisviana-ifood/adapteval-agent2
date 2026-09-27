"""Semantic analysis of conversation content"""

from typing import Dict, Any, List, Set
from pathlib import Path
from functools import lru_cache
import re

from src.shared.logger import get_logger

logger = get_logger(__name__)

_ENGLISH_WORDS_PATH = Path(__file__).parent / "resources" / "english_words.txt"


@lru_cache(maxsize=1)
def _load_english_words() -> Set[str]:
    """Load the bundled English wordlist used to keep topics English-only"""
    with open(_ENGLISH_WORDS_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


class SemanticAnalyzer:
    """Analyzes semantic content, topics, and tone"""

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Analyze semantic patterns in conversation

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with semantic metrics
        """
        # Extract key phrases
        key_phrases = self._extract_key_phrases(conversation)

        # Extract sentiment indicators
        sentiment_score = self._estimate_sentiment(conversation)

        # Topic extraction
        topics = self._extract_topics(conversation)

        # Vocabulary diversity
        vocab_diversity = self._calculate_vocabulary_diversity(
            conversation
        )

        analysis = {
            "key_phrases": key_phrases,
            "sentiment_score": sentiment_score,
            "topics": topics,
            "vocabulary_diversity": vocab_diversity,
            "total_tokens": len(conversation.split()),
            "unique_tokens": len(set(conversation.lower().split())),
        }

        logger.debug(f"Semantic analysis complete: {analysis}")
        return analysis

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract important phrases from text"""
        # Simple heuristic: phrases between 2-4 words with capitals
        words = text.split()
        phrases = []

        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            if self._is_significant_phrase(phrase):
                phrases.append(phrase)

        # Return top 10 most common
        return list(set(phrases))[:10]

    def _is_significant_phrase(self, phrase: str) -> bool:
        """Check if phrase is significant"""
        # Phrases with important keywords
        important_keywords = {
            "important",
            "critical",
            "key",
            "main",
            "goal",
            "task",
            "objective",
            "requirement",
            "constraint",
        }

        words_lower = phrase.lower().split()
        return any(kw in words_lower for kw in important_keywords)

    def _estimate_sentiment(self, text: str) -> float:
        """Simple sentiment estimation (0-1)"""
        positive_words = {
            "good",
            "great",
            "excellent",
            "positive",
            "helpful",
            "success",
            "successful",
            "working",
        }
        negative_words = {
            "bad",
            "poor",
            "negative",
            "error",
            "fail",
            "failed",
            "problem",
            "issue",
        }

        text_lower = text.lower()
        pos_count = sum(
            1 for word in positive_words if word in text_lower
        )
        neg_count = sum(
            1 for word in negative_words if word in text_lower
        )

        total = pos_count + neg_count
        if total == 0:
            return 0.5

        return pos_count / total

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text, restricted to English words"""
        # Simple topic extraction based on frequency
        words = text.lower().split()
        word_freq = {}
        english_words = _load_english_words()

        for word in words:
            cleaned = re.sub(r"[^a-z]", "", word)
            if len(cleaned) > 4 and cleaned in english_words:
                word_freq[cleaned] = word_freq.get(cleaned, 0) + 1

        # Top 5 most frequent words as topics
        sorted_words = sorted(
            word_freq.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [word for word, _ in sorted_words[:5]]

    def _calculate_vocabulary_diversity(self, text: str) -> float:
        """Calculate vocabulary diversity (0-1)"""
        words = text.lower().split()
        if not words:
            return 0.0

        unique_words = len(set(words))
        return min(unique_words / len(words), 1.0)
