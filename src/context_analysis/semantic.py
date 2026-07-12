"""Semantic analysis of conversation content"""

from typing import Dict, Any, List, Set
import re

from src.shared.logger import get_logger

logger = get_logger(__name__)


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
        """Extract main topics from text"""
        # Simple topic extraction based on frequency
        words = text.lower().split()
        word_freq = {}

        for word in words:
            if len(word) > 4:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

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
