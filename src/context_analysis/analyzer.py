"""Aggregate context analysis combining structural, semantic, complexity and intent"""

from typing import Dict, Any

from src.shared.logger import get_logger
from .structural import StructuralAnalyzer
from .semantic import SemanticAnalyzer
from .complexity import ComplexityAnalyzer
from .intent import IntentAnalyzer

logger = get_logger(__name__)


class ContextAnalyzer:
    """Runs all context analysis dimensions and combines them into one result"""

    def __init__(self):
        self.structural_analyzer = StructuralAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.intent_analyzer = IntentAnalyzer()

    def analyze(self, conversation: str) -> Dict[str, Any]:
        """
        Run structural, semantic, complexity and intent analysis

        Args:
            conversation: Raw conversation text

        Returns:
            Dict with one key per analysis dimension:
            "structural", "semantic", "complexity", "intent"
        """
        context = {
            "structural": self.structural_analyzer.analyze(conversation),
            "semantic": self.semantic_analyzer.analyze(conversation),
            "complexity": self.complexity_analyzer.analyze(conversation),
            "intent": self.intent_analyzer.analyze(conversation),
        }

        logger.debug(f"Context analysis complete: {context}")
        return context
