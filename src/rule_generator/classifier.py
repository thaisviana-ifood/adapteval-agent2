"""Task classification for specific evaluation templates"""

from typing import Dict, Any, List

from src.shared.logger import get_logger
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager

logger = get_logger(__name__)


class TaskClassifier:
    """Classify tasks and select appropriate evaluation rules"""

    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()
        self.task_types = [
            "question_answering",
            "code_generation",
            "text_summarization",
            "translation",
            "classification",
            "instruction_following",
            "reasoning",
            "creative_writing",
        ]

    def classify(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the task based on context

        Args:
            context: Context analysis results

        Returns:
            Classification results with task type and confidence
        """
        user_intents = context.get("intent", {}).get("user_intents", [])
        topics = context.get("semantic", {}).get("topics", [])

        # Simple rule-based classification
        task_type = self._rule_based_classification(
            user_intents, topics
        )

        # Get evaluation templates for this task type
        templates = self._get_templates_for_task(task_type)

        classification = {
            "task_type": task_type,
            "confidence": 0.7,
            "templates": templates,
            "recommended_evaluators": (
                self._get_evaluators_for_task(task_type)
            ),
        }

        logger.debug(
            f"Task classified as: {task_type} "
            f"(confidence: {classification['confidence']})"
        )
        return classification

    def _rule_based_classification(
        self, intents: List[str], topics: List[str]
    ) -> str:
        """Rule-based task classification"""
        keywords_map = {
            "question_answering": [
                "answer",
                "question",
                "explain",
                "clarify",
            ],
            "code_generation": [
                "code",
                "function",
                "implement",
                "algorithm",
            ],
            "text_summarization": [
                "summarize",
                "summary",
                "abstract",
                "brief",
            ],
            "instruction_following": [
                "follow",
                "instruction",
                "step",
                "process",
            ],
            "creative_writing": [
                "write",
                "create",
                "compose",
                "story",
            ],
        }

        combined = intents + topics
        for task_type, keywords in keywords_map.items():
            if any(kw in combined for kw in keywords):
                return task_type

        return "general"

    def _get_templates_for_task(self, task_type: str) -> List[str]:
        """Get evaluation templates for task type"""
        templates_map = {
            "question_answering": [
                "accuracy",
                "completeness",
                "clarity",
            ],
            "code_generation": [
                "correctness",
                "efficiency",
                "readability",
            ],
            "text_summarization": [
                "conciseness",
                "informativeness",
                "coherence",
            ],
            "instruction_following": [
                "compliance",
                "completeness",
                "correctness",
            ],
        }

        return templates_map.get(task_type, ["general", "relevance"])

    def _get_evaluators_for_task(self, task_type: str) -> List[str]:
        """Get recommended evaluators for task type"""
        evaluators_map = {
            "question_answering": [
                "factual_accuracy",
                "relevance",
                "clarity",
            ],
            "code_generation": [
                "syntax_correctness",
                "functional_correctness",
                "best_practices",
            ],
            "text_summarization": [
                "content_preservation",
                "conciseness",
                "readability",
            ],
        }

        return evaluators_map.get(task_type, ["general"])
