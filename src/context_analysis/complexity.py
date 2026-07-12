"""Complexity analysis of conversation content"""

from typing import Dict, Any

from src.shared.logger import get_logger

logger = get_logger(__name__)


class ComplexityAnalyzer:
    """Analyzes conversation complexity metrics"""

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Analyze complexity indicators

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with complexity metrics
        """
        words = conversation.split()
        sentences = conversation.split(".")

        # Calculate readability metrics
        avg_word_length = (
            sum(len(word) for word in words) / len(words)
            if words
            else 0
        )

        avg_sentence_length = (
            len(words) / len(sentences)
            if sentences
            else 0
        )

        # Estimate semantic diversity
        unique_tokens = len(set(w.lower() for w in words))
        semantic_diversity = (
            unique_tokens / len(words) if words else 0
        )

        # Count complex markers
        technical_terms = self._count_technical_terms(conversation)
        question_count = conversation.count("?")
        exclamation_count = conversation.count("!")

        complexity_score = self._calculate_complexity_score(
            avg_word_length,
            avg_sentence_length,
            semantic_diversity,
        )

        analysis = {
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "semantic_diversity": semantic_diversity,
            "technical_terms_count": technical_terms,
            "question_count": question_count,
            "exclamation_count": exclamation_count,
            "complexity_score": complexity_score,
        }

        logger.debug(f"Complexity analysis complete: {analysis}")
        return analysis

    def _count_technical_terms(self, text: str) -> int:
        """Count occurrences of technical terminology"""
        technical_keywords = {
            "algorithm",
            "function",
            "variable",
            "database",
            "api",
            "server",
            "client",
            "protocol",
            "framework",
            "library",
            "module",
            "interface",
            "architecture",
        }

        text_lower = text.lower()
        count = sum(1 for term in technical_keywords if term in text_lower)
        return count

    def _calculate_complexity_score(
        self,
        avg_word_length: float,
        avg_sentence_length: float,
        semantic_diversity: float,
    ) -> float:
        """
        Calculate overall complexity score (0-10)

        Higher score = more complex
        """
        # Normalize metrics to 0-10 scale
        word_length_score = min(avg_word_length / 0.5, 10)
        sentence_length_score = min(avg_sentence_length / 1.0, 10)
        diversity_score = semantic_diversity * 10

        # Weighted average
        score = (
            word_length_score * 0.3
            + sentence_length_score * 0.4
            + diversity_score * 0.3
        )

        return round(min(score, 10), 2)
