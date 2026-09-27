"""Tests for context analysis module"""

import pytest
from src.context_analysis import (
    StructuralAnalyzer,
    SemanticAnalyzer,
    ComplexityAnalyzer,
    IntentAnalyzer,
    ContextAnalyzer,
)


class TestStructuralAnalyzer:
    """Test structural analysis"""

    def test_analyze(self):
        """Test structural analysis"""
        analyzer = StructuralAnalyzer()
        conversation = (
            "User: Hello\n\nAssistant: Hi there!\n\n"
            "User: How are you?\n\nAssistant: I'm doing well."
        )

        result = analyzer.analyze(conversation)

        assert "total_turns" in result
        assert result["total_turns"] > 0
        assert "user_turns" in result
        assert "assistant_turns" in result


class TestSemanticAnalyzer:
    """Test semantic analysis"""

    def test_analyze(self):
        """Test semantic analysis"""
        analyzer = SemanticAnalyzer()
        text = (
            "This is an important test about machine learning "
            "and algorithms"
        )

        result = analyzer.analyze(text)

        assert "key_phrases" in result
        assert "sentiment_score" in result
        assert "topics" in result
        assert "vocabulary_diversity" in result


class TestComplexityAnalyzer:
    """Test complexity analysis"""

    def test_analyze(self):
        """Test complexity analysis"""
        analyzer = ComplexityAnalyzer()
        text = (
            "Complex technical systems require careful consideration "
            "and algorithmic optimization."
        )

        result = analyzer.analyze(text)

        assert "complexity_score" in result
        assert 0 <= result["complexity_score"] <= 10
        assert "technical_terms_count" in result


class TestIntentAnalyzer:
    """Test intent analysis"""

    def test_analyze(self):
        """Test intent analysis"""
        analyzer = IntentAnalyzer()
        conversation = "Can you explain how neural networks work?"

        result = analyzer.analyze(conversation)

        assert "user_intents" in result
        assert "assistant_objectives" in result
        assert "intent_alignment_score" in result


class TestContextAnalyzer:
    """Test the combined context analyzer"""

    def test_analyze(self):
        """Test that all four dimensions are present in the result"""
        analyzer = ContextAnalyzer()
        conversation = (
            "User: Can you explain how neural networks work?\n\n"
            "Assistant: Neural networks are computing systems "
            "inspired by biological neurons."
        )

        result = analyzer.analyze(conversation)

        assert set(result.keys()) == {
            "structural",
            "semantic",
            "complexity",
            "intent",
        }
        assert "total_turns" in result["structural"]
        assert "topics" in result["semantic"]
        assert "complexity_score" in result["complexity"]
        assert "user_intents" in result["intent"]


if __name__ == "__main__":
    pytest.main([__file__])
