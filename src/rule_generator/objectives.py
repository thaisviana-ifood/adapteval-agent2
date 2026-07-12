"""Definition of success criteria and objectives"""

from typing import Dict, Any, List

from src.shared.logger import get_logger

logger = get_logger(__name__)


class ObjectiveDefinition:
    """Define success criteria and objectives for evaluation"""

    def __init__(self):
        self.objectives_map = {
            "question_answering": {
                "primary": "Provide accurate and complete answer",
                "secondary": [
                    "Clear explanation",
                    "Relevant references",
                ],
                "success_criteria": [
                    "Addresses all parts of question",
                    "Factually correct",
                    "Well-structured",
                ],
            },
            "code_generation": {
                "primary": "Generate correct and efficient code",
                "secondary": [
                    "Readable code",
                    "Proper error handling",
                    "Good documentation",
                ],
                "success_criteria": [
                    "Syntax correct",
                    "Solves the problem",
                    "No logical errors",
                ],
            },
            "text_summarization": {
                "primary": "Create concise summary preserving key info",
                "secondary": [
                    "Readability",
                    "Coherence",
                    "Proper length",
                ],
                "success_criteria": [
                    "Preserves main points",
                    "Concise",
                    "No missing critical info",
                ],
            },
            "general": {
                "primary": "Provide helpful and relevant response",
                "secondary": [
                    "Clear communication",
                    "Appropriate length",
                ],
                "success_criteria": [
                    "Addresses user need",
                    "Factually sound",
                    "Well-organized",
                ],
            },
        }

    def get_objectives(self, task_type: str) -> Dict[str, Any]:
        """
        Get objectives for a specific task type

        Args:
            task_type: Type of task being evaluated

        Returns:
            Dict with primary, secondary objectives and success criteria
        """
        objectives = self.objectives_map.get(
            task_type, self.objectives_map["general"]
        )

        logger.debug(f"Retrieved objectives for task type: {task_type}")
        return objectives

    def define_success(
        self, task_type: str, additional_criteria: List[str] = None
    ) -> Dict[str, Any]:
        """
        Define success criteria for evaluation

        Args:
            task_type: Type of task
            additional_criteria: Extra criteria to add

        Returns:
            Complete success definition
        """
        objectives = self.get_objectives(task_type)

        success_def = {
            "primary_objective": objectives["primary"],
            "secondary_objectives": objectives["secondary"],
            "must_haves": objectives["success_criteria"],
            "nice_to_haves": additional_criteria or [],
            "failure_conditions": (
                self._get_failure_conditions(task_type)
            ),
        }

        logger.debug(f"Success definition for {task_type}: {success_def}")
        return success_def

    def _get_failure_conditions(self, task_type: str) -> List[str]:
        """Get conditions that would constitute failure"""
        failure_map = {
            "question_answering": [
                "Does not answer the question",
                "Contains false information",
                "Incomplete or partial answer",
                "Incoherent response",
            ],
            "code_generation": [
                "Code doesn't compile/run",
                "Logic errors present",
                "Doesn't solve the problem",
                "Unsafe or dangerous code",
            ],
            "text_summarization": [
                "Loses critical information",
                "Longer than original",
                "Incoherent",
                "Misrepresents source",
            ],
            "general": [
                "Doesn't address user need",
                "Contains false information",
                "Incomprehensible",
                "Harmful or offensive",
            ],
        }

        return failure_map.get(
            task_type, failure_map["general"]
        )

    def define_subgoals(self, task_type: str) -> List[Dict[str, Any]]:
        """
        Define intermediate subgoals/checkpoints

        Args:
            task_type: Type of task

        Returns:
            List of subgoal definitions
        """
        subgoals_map = {
            "code_generation": [
                {
                    "name": "Problem Understanding",
                    "description": "Correctly understands requirements",
                },
                {
                    "name": "Algorithm Selection",
                    "description": "Chooses appropriate algorithm",
                },
                {
                    "name": "Implementation",
                    "description": "Correctly implements the solution",
                },
                {
                    "name": "Testing",
                    "description": "Considers edge cases",
                },
            ],
            "general": [
                {
                    "name": "Understanding",
                    "description": "Understands the request",
                },
                {
                    "name": "Planning",
                    "description": "Plans the response",
                },
                {
                    "name": "Execution",
                    "description": "Executes the plan",
                },
                {
                    "name": "Verification",
                    "description": "Verifies correctness",
                },
            ],
        }

        return subgoals_map.get(task_type, subgoals_map["general"])
