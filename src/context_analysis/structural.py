"""Structural analysis of conversation patterns"""

from typing import Dict, Any, List
from itertools import groupby
import re

from src.shared.logger import get_logger

logger = get_logger(__name__)


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
        lines = conversation.split("\n")

        turn_count = 0
        user_turns = 0
        assistant_turns = 0
        speaker_pattern = []

        for line in lines:
            if line.strip().startswith("User:"):
                user_turns += 1
                speaker_pattern.append("user")
            elif line.strip().startswith("Assistant:"):
                assistant_turns += 1
                speaker_pattern.append("assistant")

        turn_count = user_turns + assistant_turns

        # Collapse consecutive repeats so the pattern reflects turn-taking,
        # not how many lines each speaker's turn happened to span
        speaker_pattern = [key for key, _ in groupby(speaker_pattern)]

        # Calculate turn lengths
        user_turn_lengths = self._extract_turn_lengths(
            conversation, "User:"
        )
        assistant_turn_lengths = self._extract_turn_lengths(
            conversation, "Assistant:"
        )

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
        }

        logger.debug(f"Structural analysis complete: {analysis}")
        return analysis

    def _extract_turn_lengths(
        self, conversation: str, marker: str
    ) -> List[int]:
        """Extract lengths of turns marked by given marker"""
        pattern = marker.replace(":", r"\:")
        turns = re.split(f"{pattern}", conversation)

        lengths = []
        for turn in turns[1:]:  # Skip the part before first marker
            lengths.append(len(turn.split()))

        return lengths
