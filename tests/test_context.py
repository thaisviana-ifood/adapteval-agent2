"""Tests for context analysis module"""

import json

import pytest
from src.context_analysis import (
    StructuralAnalyzer,
    SemanticAnalyzer,
    ComplexityAnalyzer,
    IntentAnalyzer,
    ContextAnalyzer,
)
from src.shared.llm.client import LLMClient

# One response covering every semantic classifier's keys: each classifier
# reads only the keys it asked for, so a single stub serves all three. The
# deliberately messy values (duplicate/empty entries, a capitalized label)
# also exercise SemanticAnalyzer's normalization. `topics` is absent on
# purpose -- it comes from TF-IDF, not from an LLM.
SEMANTIC_CLASSIFIER_RESPONSE = {
    "key_phrases": [
        "important test about machine learning",
        "Important Test About Machine Learning",
        "",
    ],
    "summary": "  The user tests machine learning and algorithms.  ",
    "label": "Neutral",
    "sentiment_score": 0.5,
    "confidence": 0.6,
    "rationale": "The exchange is matter-of-fact.",
}


@pytest.fixture(autouse=True)
def mock_semantic_classifiers(monkeypatch):
    """Avoid real API calls from SemanticAnalyzer's LLM classifiers"""
    monkeypatch.setattr(
        LLMClient,
        "call",
        lambda self, *args, **kwargs: json.dumps(SEMANTIC_CLASSIFIER_RESPONSE),
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

    TEXT = (
        "This is an important test about machine learning and algorithms"
    )

    def test_analyze(self):
        """Test semantic analysis"""
        analyzer = SemanticAnalyzer()

        result = analyzer.analyze(self.TEXT)

        assert "key_phrases" in result
        assert "summary" in result
        assert "sentiment_score" in result
        assert "topics" in result
        assert "vocabulary_diversity" in result
        assert result["classifier_failures"] == []

    def test_topics_are_ranked_by_tfidf(self):
        """Topics come from TF-IDF, with English stopwords filtered out"""
        analyzer = SemanticAnalyzer()

        result = analyzer.analyze(self.TEXT)

        # "this", "is", "an", "about" and "and" are stopwords; every
        # surviving term is a single lowercase word
        assert set(result["topics"]) == {
            "algorithms",
            "important",
            "learning",
            "machine",
            "test",
        }
        assert all(topic == topic.lower() for topic in result["topics"])

    def test_classifier_output_is_normalized(self):
        """Duplicates are dropped, strings stripped, the label lowercased"""
        analyzer = SemanticAnalyzer()

        result = analyzer.analyze(self.TEXT)

        assert result["key_phrases"] == [
            "important test about machine learning"
        ]
        assert result["summary"] == (
            "The user tests machine learning and algorithms."
        )
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.5
        assert result["sentiment_confidence"] == 0.6

    def test_degrades_when_classifiers_fail(self, monkeypatch):
        """A failing classifier leaves its field empty, not the pipeline broken"""
        monkeypatch.setattr(
            LLMClient,
            "call",
            lambda self, *args, **kwargs: "not json at all",
        )
        analyzer = SemanticAnalyzer()

        result = analyzer.analyze(self.TEXT)

        assert result["key_phrases"] == []
        assert result["summary"] == ""
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.5
        assert len(result["classifier_failures"]) == 3
        # The statistical fields never depend on an LLM call
        assert result["topics"]
        assert result["total_tokens"] == len(self.TEXT.split())
        assert result["vocabulary_diversity"] > 0


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
