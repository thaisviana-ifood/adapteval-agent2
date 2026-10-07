"""Structural analysis of conversation patterns"""

from typing import Dict, Any, List
from itertools import groupby
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.shared.emoji import count_emoji_frequency
from src.shared.logger import get_logger
from src.shared.utils import preprocess_text

logger = get_logger(__name__)

# One turn per match: a speaker marker followed by everything up to the
# next marker (or end of conversation).
_TURN_PATTERN = re.compile(
    r"(User:|Assistant:)\s*(.*?)(?=(?:User:|Assistant:)|\Z)", re.DOTALL
)


class StructuralAnalyzer:
    """Analyzes conversation structure and turn sequences"""

    def __init__(self):
        self.turn_count = 0
        self.speaker_pattern = []

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Analyze structural patterns in conversation

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with structural metrics
        """
        turns = self._extract_speaker_turns(conversation)

        raw_pattern = [turn["speaker"] for turn in turns]
        user_turns = sum(1 for s in raw_pattern if s == "user")
        assistant_turns = sum(1 for s in raw_pattern if s == "assistant")
        turn_count = len(turns)

        # Collapse consecutive repeats so the pattern reflects turn-taking,
        # not how many lines each speaker's turn happened to span
        speaker_pattern = [key for key, _ in groupby(raw_pattern)]

        user_texts = [t["text"] for t in turns if t["speaker"] == "user"]
        assistant_texts = [
            t["text"] for t in turns if t["speaker"] == "assistant"
        ]

        user_turn_lengths = [len(text.split()) for text in user_texts]
        assistant_turn_lengths = [
            len(text.split()) for text in assistant_texts
        ]

        user_texts_clean = [preprocess_text(t) for t in user_texts]
        assistant_texts_clean = [
            preprocess_text(t) for t in assistant_texts
        ]

        analysis = {
            "total_turns": turn_count,
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "speaker_pattern": speaker_pattern,
            "avg_user_turn_length": (
                sum(user_turn_lengths) / len(user_turn_lengths)
                if user_turn_lengths
                else 0
            ),
            "avg_assistant_turn_length": (
                sum(assistant_turn_lengths) / len(assistant_turn_lengths)
                if assistant_turn_lengths
                else 0
            ),
            "turn_distribution_ratio": (
                user_turns / assistant_turns
                if assistant_turns > 0
                else 0
            ),
            "consecutive_speaker_stats": self._count_consecutive_runs(
                raw_pattern
            ),
            "user_vocabulary": self._extract_vocabulary(user_texts_clean),
            "assistant_vocabulary": self._extract_vocabulary(
                assistant_texts_clean
            ),
            "user_top_tfidf_terms": self._top_tfidf_terms(
                user_texts_clean
            ),
            "assistant_top_tfidf_terms": self._top_tfidf_terms(
                assistant_texts_clean
            ),
            "user_emoji_analysis": self._analyze_emojis(user_texts),
            "assistant_emoji_analysis": self._analyze_emojis(
                assistant_texts
            ),
        }

        logger.debug(f"Structural analysis complete: {analysis}")
        return analysis

    def _extract_speaker_turns(
        self, conversation: str
    ) -> List[Dict[str, str]]:
        """Split conversation into ordered (speaker, text) turns"""
        turns = []
        for match in _TURN_PATTERN.finditer(conversation):
            speaker = "user" if match.group(1) == "User:" else "assistant"
            text = match.group(2).strip()
            turns.append({"speaker": speaker, "text": text})
        return turns

    def _count_consecutive_runs(
        self, raw_pattern: List[str]
    ) -> Dict[str, Any]:
        """Count runs of consecutive messages from the same speaker"""
        runs = [
            (speaker, len(list(group)))
            for speaker, group in groupby(raw_pattern)
        ]
        run_lengths = [length for _, length in runs]
        user_run_lengths = [
            length for speaker, length in runs if speaker == "user"
        ]
        assistant_run_lengths = [
            length for speaker, length in runs if speaker == "assistant"
        ]

        return {
            "run_lengths": run_lengths,
            "max_consecutive_same_speaker": (
                max(run_lengths) if run_lengths else 0
            ),
            "max_consecutive_user": (
                max(user_run_lengths) if user_run_lengths else 0
            ),
            "max_consecutive_assistant": (
                max(assistant_run_lengths) if assistant_run_lengths else 0
            ),
            "consecutive_runs_count": sum(
                1 for length in run_lengths if length > 1
            ),
        }

    def _extract_vocabulary(self, texts: List[str]) -> List[str]:
        """Extract the vocabulary of a set of turns via CountVectorizer"""
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return []

        vectorizer = CountVectorizer()
        vectorizer.fit(non_empty)
        return list(vectorizer.get_feature_names_out())

    def _top_tfidf_terms(
        self, texts: List[str], top_n: int = 10
    ) -> List[str]:
        """Rank terms by mean TF-IDF score across a set of turns"""
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return []

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(non_empty)
        feature_names = vectorizer.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1

        ranked = sorted(
            zip(feature_names, mean_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [term for term, _ in ranked[:top_n]]

    def _analyze_emojis(self, texts: List[str]) -> Dict[str, Any]:
        """Count emoji usage within a set of turns"""
        frequency = count_emoji_frequency(texts)

        return {
            "total_emoji_count": sum(frequency.values()),
            "unique_emoji_count": len(frequency),
            "emoji_frequency": frequency,
        }
