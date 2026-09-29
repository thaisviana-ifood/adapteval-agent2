"""Evaluator implementations for jury assessment"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.config import JURY_LLM_MAX_TOKENS, JURY_PROVIDERS
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
        client_kwargs: Dict[str, Any] = {"max_tokens": JURY_LLM_MAX_TOKENS}
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
        Evaluate response using LLM, as a True/False verdict per criterion

        Args:
            response: Response to evaluate
            criteria: Evaluation criteria
            context: Context information

        Returns:
            Evaluation results with per-criterion verdicts and reasoning
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

        criterion_names = [
            c["name"] for c in criteria.get("weighted_criteria", [])
        ]
        evaluation = self._parse_evaluation(eval_text, criterion_names)
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

    def _parse_evaluation(
        self, eval_text: str, criterion_names: List[str]
    ) -> Dict[str, Any]:
        """Parse the LLM's per-criterion True/False verdicts

        A criterion the model didn't answer in the expected
        "<name>: True|False" format is simply left out of "verdicts" --
        this juror abstains on it rather than defaulting to a guessed
        verdict, so the panel's per-criterion vote isn't skewed by a
        parsing failure.
        """
        # Simple parsing - can be enhanced
        evaluation: Dict[str, Any] = {
            "verdicts": {},
            "confidence": 0.5,
            "reasoning": eval_text,
        }

        lines = eval_text.split("\n")
        for name in criterion_names:
            prefix = f"{name.lower()}:"
            for line in lines:
                stripped = line.strip()
                if not stripped.lower().startswith(prefix):
                    continue
                verdict_text = stripped.split(":", 1)[1].strip().lower()
                if verdict_text.startswith("true"):
                    evaluation["verdicts"][name] = True
                elif verdict_text.startswith("false"):
                    evaluation["verdicts"][name] = False
                break

        for line in lines:
            if line.strip().lower().startswith("confidence:"):
                try:
                    evaluation["confidence"] = float(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    pass
                break

        return evaluation

    def get_name(self) -> str:
        """Get evaluator name"""
        return f"LLM-{self.evaluator_type}"


class TypesafeEvaluator(Evaluator):
    """LLM-as-judge evaluator backed by the Typesafe structured Q&A API

    Typesafe isn't OpenAI-compatible: instead of a free-text prompt that
    gets regex-parsed (like LLMEvaluator), it takes typed questions and
    returns structured answers directly -- see TypesafeClient and
    https://docs.typesafe.ai/introduction/quickstart. One boolean "noul"
    (yes/no) question is asked per rubric criterion.
    """

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
        """Evaluate response via one Typesafe 'noul' (yes/no) question per criterion

        Instructions mirror LLMEvaluator's "llm_evaluation" prompt (same
        task context, same "fully satisfies, no partial credit" standard) so
        Typesafe isn't held to a looser or stricter bar than the other
        jurors just because it's a different API shape.
        """
        weighted_criteria = criteria.get("weighted_criteria", [])
        task_note = (
            f"Task type: {context.get('task_type', 'general')}; "
            f"Complexity: {context.get('complexity_score', 'unknown')}"
        )

        key_to_name = {
            f"criterion_{i}": c["name"] for i, c in enumerate(weighted_criteria)
        }
        questions = {
            key: {
                "type": "noul",
                "instructions": (
                    f"Task context -- {task_note}\n\n"
                    f"Criterion -- {c['name']}: {c['description']}\n\n"
                    "Answer True only if the response clearly and fully "
                    "satisfies this criterion; otherwise answer False. Do "
                    "not give partial credit for a partially satisfied "
                    "criterion."
                ),
            }
            for key, c in zip(key_to_name, weighted_criteria)
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

        answers = result.get("answers", {})
        verdicts: Dict[str, bool] = {}
        criterion_confidences: Dict[str, float] = {}
        for key, name in key_to_name.items():
            answer = answers.get(key, {})
            # "noul" has no separate confidence field: its own answer is
            # already a 0-1 probability, so distance from the 0.5 decision
            # boundary doubles as this verdict's confidence.
            noul = float(answer.get("noul", 0.5))
            verdicts[name] = noul >= 0.5
            criterion_confidences[name] = abs(noul - 0.5) * 2

        evaluation = {
            "evaluator": self.get_name(),
            "evaluator_type": "typesafe",
            "verdicts": verdicts,
            "criterion_confidences": criterion_confidences,
            "confidence": (
                sum(criterion_confidences.values()) / len(criterion_confidences)
                if criterion_confidences
                else 0.5
            ),
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

    Jurors don't grade a response on a scale -- each one answers every
    rubric criterion True/False (see `_vote_on_criterion`). The panel's
    verdict for each criterion is a majority vote among the jurors that
    answered it: whichever side (True/False) has more votes wins, and only
    those concordant jurors' confidences count. A tie means no real
    majority, so that criterion fails closed (counts as False).
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
        Get True/False verdicts from all panel members and vote per criterion

        Args:
            response: Response to evaluate
            criteria: Evaluation criteria
            context: Context information

        Returns:
            Combined panel evaluation: a verdict per criterion, plus the
            resulting pass rate (fraction of criteria that passed the vote)
        """
        evaluations = []

        for evaluator in self.evaluators:
            evaluation = evaluator.evaluate(
                response, criteria, context
            )
            evaluations.append(evaluation)

        criterion_names: List[str] = []
        seen = set()
        for c in criteria.get("weighted_criteria", []):
            if c["name"] not in seen:
                seen.add(c["name"])
                criterion_names.append(c["name"])

        verdicts: Dict[str, Dict[str, Any]] = {
            name: self._vote_on_criterion(name, evaluations)
            for name in criterion_names
        }

        passed_count = sum(1 for v in verdicts.values() if v["passed"])
        pass_rate = passed_count / len(verdicts) if verdicts else 0.0
        overall_confidence = (
            sum(v["confidence"] for v in verdicts.values()) / len(verdicts)
            if verdicts
            else 0.0
        )

        panel_result = {
            "evaluations": evaluations,
            "panel_size": len(evaluations),
            "verdicts": verdicts,
            "passed_count": passed_count,
            "criteria_count": len(verdicts),
            "pass_rate": pass_rate,
            "confidence": overall_confidence,
            # Backward-compat scalar (0-10) for consumers built around a
            # numeric jury score (FinalScoreCalculator, AccuracyCalculator,
            # HITL) -- derived from pass_rate, not a juror-assigned grade.
            "average_score": pass_rate * 10,
        }

        logger.debug(
            f"Panel evaluation complete. "
            f"Pass rate: {passed_count}/{len(verdicts)} criteria"
        )
        return panel_result

    def _vote_on_criterion(
        self, criterion_name: str, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Majority-vote a single criterion across jurors that answered it

        Jurors that didn't return a verdict for this criterion (parsing
        failure, question omitted, etc.) simply don't get a vote. Whichever
        side -- True or False -- has more votes wins, and only those
        concordant jurors' confidences feed the criterion's confidence. A
        tie (no real majority) fails closed: the criterion counts as False.
        """
        votes: Dict[str, bool] = {}
        confidences: Dict[str, float] = {}

        for evaluation in evaluations:
            criterion_verdicts = evaluation.get("verdicts", {})
            if criterion_name not in criterion_verdicts:
                continue
            evaluator_name = evaluation.get("evaluator", "evaluator")
            votes[evaluator_name] = criterion_verdicts[criterion_name]
            # Prefer this juror's per-criterion confidence (e.g. Typesafe's
            # noul distance-from-0.5) over its one overall confidence, which
            # would otherwise flatten every criterion to the same number.
            confidences[evaluator_name] = evaluation.get(
                "criterion_confidences", {}
            ).get(criterion_name, evaluation.get("confidence", 0.5))

        if not votes:
            return {
                "passed": False,
                "agreement": False,
                "votes": {},
                "confidence": 0.0,
                "concordant_evaluators": [],
            }

        true_voters = [name for name, v in votes.items() if v]
        false_voters = [name for name, v in votes.items() if not v]

        if len(true_voters) > len(false_voters):
            concordant, passed = true_voters, True
        elif len(false_voters) > len(true_voters):
            concordant, passed = false_voters, False
        else:
            concordant, passed = [], False

        confidence = (
            sum(confidences[name] for name in concordant) / len(concordant)
            if concordant
            else 0.0
        )

        return {
            "passed": passed,
            "agreement": len(concordant) == len(votes),
            "votes": votes,
            "confidence": confidence,
            "concordant_evaluators": concordant,
        }
