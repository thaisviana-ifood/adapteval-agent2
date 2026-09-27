"""Evaluator implementations for jury assessment"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod

from src.shared.logger import get_logger
from src.shared.llm.client import LLMClient
from src.shared.llm.prompts import PromptManager

logger = get_logger(__name__)


class Evaluator(ABC):
    """Abstract base class for evaluators"""

    @abstractmethod
    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a response"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get evaluator name"""
        pass


class LLMEvaluator(Evaluator):
    """LLM-based evaluator using Claude"""

    def __init__(self, evaluator_type: str = "general"):
        self.llm_client = LLMClient()
        self.prompt_manager = PromptManager()
        self.evaluator_type = evaluator_type

    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate response using LLM

        Args:
            response: Response to evaluate
            criteria: Evaluation criteria
            context: Context information

        Returns:
            Evaluation results with scores and reasoning
        """
        prompt = self._build_eval_prompt(
            response, criteria, context
        )

        eval_text = self.llm_client.call(
            prompt,
            name=f"jury.{self.get_name()}",
            langfuse_prompt=self.prompt_manager.get_prompt_client("llm_evaluation"),
            metadata={
                "evaluator_type": self.evaluator_type,
                "task_type": context.get("task_type", "general"),
            },
        )

        evaluation = self._parse_evaluation(eval_text)
        evaluation["evaluator"] = self.get_name()
        evaluation["evaluator_type"] = self.evaluator_type

        logger.debug(f"LLM evaluation complete: {evaluation}")
        return evaluation

    def _build_eval_prompt(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Build evaluation prompt"""
        criteria_text = "\n".join(
            [
                f"- {c['name']}: {c['description']}"
                for c in criteria.get("weighted_criteria", [])
            ]
        )

        return self.prompt_manager.format_prompt(
            "llm_evaluation",
            response=response,
            criteria_text=criteria_text,
            task_type=context.get("task_type", "general"),
            complexity_score=context.get("complexity_score", "unknown"),
        )

    def _parse_evaluation(self, eval_text: str) -> Dict[str, Any]:
        """Parse LLM evaluation output"""
        # Simple parsing - can be enhanced
        evaluation = {
            "score": 5.0,
            "confidence": 0.5,
            "strengths": [],
            "weaknesses": [],
            "reasoning": eval_text,
        }

        # Try to extract score
        if "Score" in eval_text:
            lines = eval_text.split("\n")
            for line in lines:
                if "Score" in line:
                    try:
                        score_str = line.split(":")[-1].strip()
                        evaluation["score"] = float(
                            score_str.split("/")[0]
                        )
                    except ValueError:
                        pass

        return evaluation

    def get_name(self) -> str:
        """Get evaluator name"""
        return f"LLM-{self.evaluator_type}"


class RulesBasedEvaluator(Evaluator):
    """Rules-based evaluator"""

    def __init__(self):
        self.name = "RulesBased"

    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate response using rules

        Args:
            response: Response to evaluate
            criteria: Evaluation criteria
            context: Context information

        Returns:
            Evaluation results
        """
        scores = self._apply_rules(response, criteria, context)

        evaluation = {
            "evaluator": self.get_name(),
            "scores": scores,
            "average_score": sum(scores.values()) / len(scores)
            if scores
            else 0,
            "confidence": 0.6,
        }

        logger.debug(f"Rules-based evaluation complete: {evaluation}")
        return evaluation

    def _apply_rules(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Apply rules to evaluate"""
        scores = {}

        for criterion in criteria.get("weighted_criteria", []):
            score = self._score_criterion(
                response, criterion, context
            )
            scores[criterion["name"]] = score

        return scores

    def _score_criterion(
        self,
        response: str,
        criterion: Dict[str, Any],
        context: Dict[str, Any],
    ) -> float:
        """Score a single criterion"""
        # Simple heuristic scoring
        if criterion["name"] == "Relevance":
            return self._score_relevance(response, context)
        elif criterion["name"] == "Correctness":
            return self._score_correctness(response, context)
        elif criterion["name"] == "Clarity":
            return self._score_clarity(response)
        else:
            return 5.0

    def _score_relevance(self, response: str, context: Dict) -> float:
        """Score relevance"""
        if len(response) < 10:
            return 2.0
        return min(len(response) / 500, 10.0)

    def _score_correctness(
        self, response: str, context: Dict
    ) -> float:
        """Score correctness"""
        # Check for error indicators
        error_indicators = [
            "error",
            "fail",
            "incorrect",
            "wrong",
        ]
        if any(
            indicator in response.lower()
            for indicator in error_indicators
        ):
            return 4.0
        return 7.0

    def _score_clarity(self, response: str) -> float:
        """Score clarity"""
        sentences = response.split(".")
        avg_sentence_length = (
            len(response) / len(sentences)
            if sentences
            else 0
        )

        if avg_sentence_length > 150:
            return 4.0
        elif avg_sentence_length < 20:
            return 6.0
        else:
            return 8.0

    def get_name(self) -> str:
        """Get evaluator name"""
        return self.name


class EvaluatorPanel:
    """Panel of multiple evaluators"""

    def __init__(self, evaluator_count: int = 3):
        self.evaluators: List[Evaluator] = []
        self._init_evaluators(evaluator_count)

    def _init_evaluators(self, count: int) -> None:
        """Initialize evaluators"""
        self.evaluators.append(RulesBasedEvaluator())

        for i in range(count - 1):
            self.evaluators.append(
                LLMEvaluator(evaluator_type=f"type_{i}")
            )

    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get evaluations from all panel members

        Args:
            response: Response to evaluate
            criteria: Evaluation criteria
            context: Context information

        Returns:
            Combined panel evaluation
        """
        evaluations = []

        for evaluator in self.evaluators:
            evaluation = evaluator.evaluate(
                response, criteria, context
            )
            evaluations.append(evaluation)

        panel_result = {
            "evaluations": evaluations,
            "panel_size": len(evaluations),
            "average_score": (
                sum(
                    e.get("score", e.get("average_score", 5))
                    for e in evaluations
                )
                / len(evaluations)
            ),
            "confidence": (
                sum(
                    e.get("confidence", 0.5)
                    for e in evaluations
                )
                / len(evaluations)
            ),
        }

        logger.debug(
            f"Panel evaluation complete. "
            f"Average score: {panel_result['average_score']:.2f}"
        )
        return panel_result
