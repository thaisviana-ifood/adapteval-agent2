"""Evaluator implementations for jury assessment"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.config import JURY_PROVIDERS
from src.shared.logger import get_logger
from src.shared.llm.client import LLMClient
from src.shared.llm.langfuse_client import get_langfuse_client
from src.shared.llm.prompts import PromptManager
from src.shared.llm.typesafe_client import TypesafeClient

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
    """LLM-based evaluator, bound to a single provider's model

    The provider (model, api_key, base_url) is passed in by whoever
    orchestrates the jury -- see EvaluatorPanel -- instead of being
    hardcoded here, so each juror can be wired to a different LLM vendor.
    """

    def __init__(
        self,
        evaluator_type: str = "general",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # `is not None` (not plain truthiness): an explicitly configured
        # empty string -- e.g. an unset TYPESAFE_BASE_URL -- must still be
        # passed through, so the client fails loudly instead of silently
        # falling back to LLMClient's DeepSeek-flavored defaults.
        client_kwargs: Dict[str, str] = {}
        if model is not None:
            client_kwargs["model"] = model
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self.llm_client = LLMClient(**client_kwargs)
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


class TypesafeEvaluator(Evaluator):
    """LLM-as-judge evaluator backed by the Typesafe structured Q&A API

    Typesafe isn't OpenAI-compatible: instead of a free-text prompt that
    gets regex-parsed for a score (like LLMEvaluator), it takes a single
    typed "score" question and returns the score/confidence directly -- see
    TypesafeClient and https://docs.typesafe.ai/introduction/quickstart.
    """

    # Typesafe's "score" question returns the index of the matching label,
    # and caps a question at 10 levels -- so this is a 0-9 scale, rescaled
    # to 0-10 below to stay comparable with the other (LLMEvaluator-based)
    # jurors' scores.
    _SCALE_LABELS = [
        "Completely fails the criteria",
        "Very poor",
        "Poor",
        "Below average",
        "Slightly below average",
        "Average",
        "Slightly above average",
        "Good",
        "Very good",
        "Outstanding, fully meets all criteria",
    ]

    def __init__(self, model: str, api_key: str, base_url: str, timeout: int = 60):
        self.client = TypesafeClient(
            model=model, api_key=api_key, base_url=base_url, timeout=timeout
        )
        # TypesafeClient is plain httpx, not the langfuse.openai wrapper
        # LLMClient uses, so calls through it aren't auto-traced -- trace
        # them manually here to keep this juror visible in Langfuse
        # alongside the DeepSeek/OpenAI jurors.
        self._langfuse = get_langfuse_client()

    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate response via a single Typesafe 'score' question"""
        criteria_text = "\n".join(
            f"- {c['name']}: {c['description']}"
            for c in criteria.get("weighted_criteria", [])
        )
        instructions = (
            "Rate the overall quality of the response against these criteria:\n"
            f"{criteria_text}\n\n"
            f"Task type: {context.get('task_type', 'general')}; "
            f"Complexity: {context.get('complexity_score', 'unknown')}"
        )
        questions = {
            "overall_quality": {
                "type": "score",
                "instructions": instructions,
                "criteria": self._SCALE_LABELS,
            }
        }

        if self._langfuse is not None:
            with self._langfuse.start_as_current_observation(
                as_type="generation",
                name=f"jury.{self.get_name()}",
                model=self.client.model,
                input={"state": response, "questions": questions},
                metadata={
                    "evaluator_type": "typesafe",
                    "task_type": context.get("task_type", "general"),
                },
            ) as generation:
                result = self.client.ask(state=response, questions=questions)
                generation.update(
                    output=result.get("answers"),
                    usage_details=result.get("usage"),
                )
        else:
            result = self.client.ask(state=response, questions=questions)

        answer = result.get("answers", {}).get("overall_quality", {})
        raw_score = float(answer.get("score", 4.5))  # 0-9
        legend = answer.get("legend", {})

        evaluation = {
            "evaluator": self.get_name(),
            "evaluator_type": "typesafe",
            "score": raw_score * 10 / (len(self._SCALE_LABELS) - 1),  # rescaled to 0-10
            "confidence": float(answer.get("confidence", 0.5)),
            "reasoning": legend.get(str(int(raw_score)), ""),
        }

        logger.debug(f"Typesafe evaluation complete: {evaluation}")
        return evaluation

    def get_name(self) -> str:
        """Get evaluator name"""
        return "LLM-typesafe"


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
    """Orchestrates the jury: one LLM juror per configured provider

    Each entry in `providers` supplies the (name, model, api_key, base_url)
    for one juror; they're passed straight through as parameters to
    LLMEvaluator, so adding/swapping a provider is a config change, not a
    code change. Defaults to JURY_PROVIDERS (DeepSeek, OpenAI, Typesafe --
    see src/config.py), matching MIN_EVALUATORS=3.
    """

    def __init__(self, providers: Optional[List[Dict[str, str]]] = None):
        self.evaluators: List[Evaluator] = []
        self._init_evaluators(providers or JURY_PROVIDERS)

    def _init_evaluators(self, providers: List[Dict[str, str]]) -> None:
        """Initialize one juror per provider

        Typesafe isn't OpenAI-compatible, so it gets TypesafeEvaluator;
        every other provider goes through the generic LLMEvaluator.
        """
        for provider in providers:
            if provider["name"] == "typesafe":
                self.evaluators.append(
                    TypesafeEvaluator(
                        model=provider["model"],
                        api_key=provider["api_key"],
                        base_url=provider["base_url"],
                    )
                )
            else:
                self.evaluators.append(
                    LLMEvaluator(
                        evaluator_type=provider["name"],
                        model=provider.get("model"),
                        api_key=provider.get("api_key"),
                        base_url=provider.get("base_url"),
                    )
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
