"""Intent analysis for conversation understanding"""

from typing import Dict, Any, List

from src.shared.logger import get_logger

logger = get_logger(__name__)


class IntentAnalyzer:
    """Analyzes user and assistant intents in conversation"""

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Analyze intents in conversation

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with intent analysis
        """
        # Extract user intents
        user_intents = self._extract_user_intents(conversation)

        # Extract assistant objectives
        assistant_objectives = self._extract_assistant_objectives(
            conversation
        )

        # Analyze intent alignment
        alignment_score = self._calculate_intent_alignment(
            user_intents, assistant_objectives
        )

        # Segmentation by intent
        intent_segments = self._segment_by_intent(conversation)

        analysis = {
            "user_intents": user_intents,
            "assistant_objectives": assistant_objectives,
            "intent_alignment_score": alignment_score,
            "intent_segments": intent_segments,
            "primary_intent": (
                user_intents[0] if user_intents else "unknown"
            ),
        }

        logger.debug(f"Intent analysis complete: {analysis}")
        return analysis

    def _extract_user_intents(self, text: str) -> List[str]:
        """Extract user intentions from conversation"""
        intent_keywords = {
            "help": ["help", "assist", "guide", "support"],
            "explain": ["explain", "clarify", "understand", "how"],
            "fix": ["fix", "bug", "error", "issue", "problem"],
            "create": ["create", "build", "make", "develop"],
            "analyze": [
                "analyze",
                "review",
                "evaluate",
                "check",
                "inspect",
            ],
            "learn": ["learn", "teach", "tutorial", "guide"],
            "optimize": [
                "optimize",
                "improve",
                "better",
                "faster",
                "efficient",
            ],
        }

        detected_intents = []
        text_lower = text.lower()

        for intent, keywords in intent_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_intents.append(intent)

        return detected_intents if detected_intents else ["general_inquiry"]

    def _extract_assistant_objectives(self, text: str) -> List[str]:
        """Extract assistant's stated objectives"""
        objective_keywords = {
            "inform": [
                "explain",
                "inform",
                "provide",
                "describe",
                "tell",
            ],
            "assist": ["help", "assist", "guide", "support", "enable"],
            "validate": ["validate", "check", "verify", "confirm"],
            "suggest": ["suggest", "recommend", "propose", "advise"],
            "implement": [
                "implement",
                "create",
                "build",
                "develop",
                "write",
            ],
        }

        detected_objectives = []
        text_lower = text.lower()

        for objective, keywords in objective_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_objectives.append(objective)

        return (
            detected_objectives
            if detected_objectives
            else ["provide_assistance"]
        )

    def _calculate_intent_alignment(
        self,
        user_intents: List[str],
        assistant_objectives: List[str],
    ) -> float:
        """
        Calculate how well assistant objectives align with user intents

        Returns:
            Score from 0-1
        """
        if not user_intents or not assistant_objectives:
            return 0.5

        # Simple overlap-based alignment
        overlap = len(set(user_intents) & set(assistant_objectives))
        max_possible = max(len(user_intents), len(assistant_objectives))

        return overlap / max_possible if max_possible > 0 else 0.5

    def _segment_by_intent(self, conversation: str) -> Dict[str, str]:
        """Segment conversation by detected intents"""
        segments = {}
        current_intent = "unknown"

        turns = conversation.split("\n\n")
        for i, turn in enumerate(turns):
            intent = self._detect_turn_intent(turn)
            if intent not in segments:
                segments[intent] = ""
            segments[intent] += turn + "\n"

        return segments

    def _detect_turn_intent(self, turn: str) -> str:
        """Detect intent of a single turn"""
        keywords = {
            "question": ["?", "what", "how", "why", "when", "where"],
            "request": ["please", "could", "would", "can you"],
            "statement": ["believe", "think", "suggest", "recommend"],
        }

        turn_lower = turn.lower()
        for intent, kws in keywords.items():
            if any(kw in turn_lower for kw in kws):
                return intent

        return "neutral"
