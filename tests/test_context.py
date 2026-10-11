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
from src.context_analysis.intent import (
    _FALLBACK_OBJECTIVE_KEYWORDS,
    _OBJECTIVE_COMPATIBILITY,
    INTENT_VOCABULARY,
    _speaker_texts,
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


def stub_llm(monkeypatch, user=None, objectives=None, alignment=None):
    """Answer each intent classifier with its own canned response

    IntentAnalyzer's three prompts are routed apart by a phrase only the
    prompt in question contains, so one stub can serve all of them while
    still letting a test assert on one field at a time. A classifier with no
    response given gets an empty object, which the analyzer treats as a
    failure and degrades from.
    """

    def call(self, prompt, *args, **kwargs):
        if "alignment_score" in prompt:
            return json.dumps(alignment or {})
        if "assistant_objectives" in prompt:
            return json.dumps(objectives or {})
        return json.dumps(user or {})

    monkeypatch.setattr(LLMClient, "call", call)


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

    CONVERSATION = (
        "User: my payment keeps failing, can you fix it?\n\n"
        "Assistant: Let me check your card details and sort that out."
    )

    def test_analyze(self):
        """Test intent analysis"""
        analyzer = IntentAnalyzer()
        conversation = "Can you explain how neural networks work?"

        result = analyzer.analyze(conversation)

        assert "user_intents" in result
        assert "assistant_objectives" in result
        assert "intent_alignment_score" in result

    def test_semantic_path_reads_labels_evidence_and_alignment(
        self, monkeypatch
    ):
        """Each classifier's output lands in the field that asked for it"""
        stub_llm(
            monkeypatch,
            user={
                "user_intents": [
                    {
                        "label": "Fix",
                        "evidence": "my payment keeps failing",
                        "confidence": 0.9,
                    },
                    {"label": "explain", "evidence": "", "confidence": 0.3},
                ],
                "primary_intent": "fix",
                "confidence": 0.88,
            },
            objectives={
                "assistant_objectives": ["verify", "implement"],
                "confidence": 0.7,
            },
            alignment={
                "alignment_score": 0.9,
                "unmet_intents": ["explain"],
                "rationale": "The assistant went after the payment failure.",
            },
        )

        result = IntentAnalyzer().analyze(self.CONVERSATION)

        # "Fix" is normalized, and the primary intent leads the list
        assert result["user_intents"] == ["fix", "explain"]
        assert result["primary_intent"] == "fix"
        assert result["primary_intent_confidence"] == 0.88
        # Segments are the classifier's evidence quotes, keyed by intent
        assert result["intent_segments"] == {
            "fix": "my payment keeps failing"
        }
        # details stay positionally parallel to user_intents
        assert [d["label"] for d in result["user_intent_details"]] == [
            "fix",
            "explain",
        ]
        assert result["assistant_objectives"] == ["verify", "implement"]
        assert result["intent_alignment_score"] == 0.9
        assert result["unmet_intents"] == ["explain"]
        assert result["classifier_failures"] == []

    def test_closed_set_mode_drops_off_taxonomy_labels(self, monkeypatch):
        """An invented label is a classifier failure, not a new intent"""
        stub_llm(
            monkeypatch,
            user={
                "user_intents": [
                    {"label": "cancel_order", "evidence": "cancel", "confidence": 1.0},
                    {"label": "something_invented", "evidence": "x", "confidence": 1.0},
                ],
                "primary_intent": "cancel_order",
                "confidence": 0.95,
            },
        )

        analyzer = IntentAnalyzer(taxonomy=["cancel_order", "track_order"])
        result = analyzer.analyze("I want to cancel my order")

        assert result["user_intents"] == ["cancel_order"]
        assert result["primary_intent"] == "cancel_order"

    def test_unusable_output_reports_unknown_instead_of_guessing(
        self, monkeypatch
    ):
        """There is no deterministic intent classifier to fall back on, so a
        broken response is reported as a failure, not papered over"""
        monkeypatch.setattr(
            LLMClient, "call", lambda self, *a, **kw: "not json at all"
        )

        result = IntentAnalyzer().analyze(self.CONVERSATION)

        assert "intent_user_classification" in result["classifier_failures"]
        assert result["user_intents"] == []
        assert result["user_intent_details"] == []
        assert result["intent_segments"] == {}
        assert result["primary_intent"] == "unknown"
        assert result["primary_intent_confidence"] == 0.0

    def test_an_all_off_taxonomy_answer_is_a_failure(self, monkeypatch):
        """A parseable response whose every label is rejected leaves no
        intent, and `call` cannot have recorded that itself"""
        stub_llm(
            monkeypatch,
            user={
                "user_intents": [{"label": "invented", "confidence": 1.0}],
                "primary_intent": "invented",
                "confidence": 0.9,
            },
        )

        result = IntentAnalyzer(taxonomy=["cancel_order"]).analyze(
            "I want to cancel my order"
        )

        assert result["primary_intent"] == "unknown"
        assert result["classifier_failures"].count(
            "intent_user_classification_closed_set"
        ) == 1

    def test_the_other_fields_survive_an_intent_failure(self, monkeypatch):
        """Objectives keep their own keyword fallback; alignment has nothing
        to align against and says so rather than scoring 0.0"""
        monkeypatch.setattr(
            LLMClient, "call", lambda self, *a, **kw: "not json at all"
        )

        result = IntentAnalyzer().analyze(self.CONVERSATION)

        assert result["assistant_objectives"]  # keyword path found objectives
        assert result["intent_alignment_score"] == 0.5
        assert "Not enough signal" in result["intent_alignment_rationale"]

    def test_fallback_alignment_is_not_zero_for_compatible_objectives(
        self, monkeypatch
    ):
        """Regression: the old implementation intersected two vocabularies
        that share no words, so this score was always 0.0

        The intent classifier answers; the objectives and alignment ones get
        an empty object, so both degrade -- which is the only way to reach
        the fallback alignment now.
        """
        stub_llm(
            monkeypatch,
            user={
                "user_intents": [
                    {"label": "fix", "evidence": "payment keeps failing"}
                ],
                "primary_intent": "fix",
                "confidence": 0.9,
            },
        )

        result = IntentAnalyzer().analyze(self.CONVERSATION)

        assert result["user_intents"] == ["fix"]
        assert result["assistant_objectives"]  # keyword path found objectives
        assert result["intent_alignment_score"] > 0.0
        assert result["unmet_intents"] == []

    def test_single_utterance_skips_assistant_classifiers(self, monkeypatch):
        """A bare utterance has no assistant turn to read or align with"""
        calls = []

        def record(self, prompt, *args, **kwargs):
            calls.append(kwargs.get("name"))
            return json.dumps(
                {
                    "user_intents": [
                        {"label": "fix", "evidence": "broken", "confidence": 1.0}
                    ],
                    "primary_intent": "fix",
                    "confidence": 1.0,
                }
            )

        monkeypatch.setattr(LLMClient, "call", record)

        result = IntentAnalyzer().analyze("my account is broken")

        assert calls == ["context_analysis.intent_user"]
        assert result["assistant_objectives"] == []
        assert result["intent_alignment_score"] == 0.5
        assert "No assistant turn" in result["intent_alignment_rationale"]


class TestObjectiveCompatibility:
    """Guard the fallback alignment map against the bug it replaced"""

    def test_every_intent_can_be_matched_by_the_fallback_vocabulary(self):
        """The fallback can only emit five objectives; every intent's set
        must reach at least one of them, or alignment is unreachable for
        that intent -- which is exactly how the old implementation failed"""
        fallback_objectives = set(_FALLBACK_OBJECTIVE_KEYWORDS)

        unreachable = {
            intent
            for intent, compatible in _OBJECTIVE_COMPATIBILITY.items()
            if not (compatible & fallback_objectives)
        }

        assert not unreachable

    def test_every_vocabulary_intent_has_a_compatibility_entry(self):
        """An intent the classifier is told to emit but this map cannot
        score would always come back as unmet"""
        assert set(INTENT_VOCABULARY) <= set(_OBJECTIVE_COMPATIBILITY)

    def test_vocabulary_labels_are_single_comparable_tokens(self):
        """rule_generator/classifier.py matches these by exact equality"""
        for label in INTENT_VOCABULARY:
            assert label == label.lower()
            assert " " not in label


class TestSpeakerSplitting:
    """Test how roles are read out of a conversation"""

    def test_explicit_markers(self):
        user, assistant = _speaker_texts(
            "User: hello\n\nAssistant: hi\n\nUser: fix it"
        )

        assert user == "hello\n\nfix it"
        assert assistant == "hi"

    def test_unmarked_text_belongs_to_the_user(self):
        """An unlabeled utterance is something the user said

        shared.utils.parse_conversation_turns() defaults an unlabeled turn to
        "assistant", which would invent an assistant turn for a bare
        utterance; intent analysis must not inherit that.
        """
        user, assistant = _speaker_texts("I want to cancel my order")

        assert user == "I want to cancel my order"
        assert assistant == ""


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
