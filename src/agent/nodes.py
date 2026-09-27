"""LangGraph node implementations for the thesis evaluation pipeline

Each node wraps the existing business-logic components (context_analysis,
rule_generator, jury, metrics) and writes its result into a distinct state
key, so the checkpointer persists the output of every stage.
"""

from typing import Any, Dict, List, Tuple

from src.context_analysis import ContextAnalyzer
from src.rule_generator import (
    TaskClassifier,
    ObjectiveDefinition,
    CriteriaGenerator,
    HeuristicChecker,
)
from src.jury import EvaluatorPanel, AccuracyCalculator, HumanInTheLoop
from src.metrics import FinalScoreCalculator
from src.shared.logger import get_logger
from src.shared.utils import parse_conversation_turns
from src.agent.schema import EvaluationOutput, MetricaAvaliada, RelatorioConsolidacao
from src.agent.state import PipelineState

logger = get_logger(__name__)


def _extract_response_and_query(conversation: str) -> Tuple[str, str]:
    """Pull the last assistant turn (the response to evaluate) and the user
    turn that preceded it (the query it answers) out of the conversation"""
    turns = parse_conversation_turns(conversation)

    response = ""
    query = ""
    for i in range(len(turns) - 1, -1, -1):
        if turns[i]["role"] == "assistant":
            response = turns[i]["content"]
            for j in range(i - 1, -1, -1):
                if turns[j]["role"] == "user":
                    query = turns[j]["content"]
                    break
            break

    return response, query


def _build_evaluation_output(
    jury_result: Dict[str, Any],
    heuristics: Dict[str, Any],
    accuracy: Dict[str, Any],
    final_score: Dict[str, Any],
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the pipeline's final structured output: avaliação (métricas,
    valores, confiança), nota de confiança global e relatório de
    consolidação das métricas"""
    metrics: List[MetricaAvaliada] = []

    for evaluation in jury_result.get("evaluations", []):
        confidence = float(evaluation.get("confidence", 0.5))
        evaluator = evaluation.get("evaluator", "evaluator")

        if "scores" in evaluation:
            for criterion, score in evaluation["scores"].items():
                metrics.append(
                    MetricaAvaliada(
                        metrica=f"{evaluator}.{criterion}",
                        valor=float(score),
                        confianca=confidence,
                    )
                )
        elif "score" in evaluation:
            metrics.append(
                MetricaAvaliada(
                    metrica=evaluator,
                    valor=float(evaluation["score"]),
                    confianca=confidence,
                )
            )

    component_confidences = {
        "jury": jury_result.get("confidence", 0.5),
        "heuristic": heuristics.get("pass_rate", 0.5),
        "accuracy": accuracy.get("confidence", 0.5),
        "context": final_score.get("overall_confidence", 0.5),
    }
    for component, data in final_score.get("component_breakdown", {}).items():
        metrics.append(
            MetricaAvaliada(
                metrica=f"final.{component}",
                valor=float(data.get("score", 5.0)),
                confianca=float(component_confidences.get(component, 0.5)),
            )
        )

    relatorio = RelatorioConsolidacao(
        evaluation_id=report.get("evaluation_id", ""),
        final_score=final_score["final_score"],
        confidence=final_score["overall_confidence"],
        recommendation=report.get("recommendation", ""),
        summary=report.get("summary", {}),
        detailed_breakdown=report.get("detailed_breakdown", {}),
    )

    output = EvaluationOutput(
        avaliacao=metrics,
        nota_confianca_global=final_score["overall_confidence"],
        relatorio_consolidacao_metricas=relatorio,
    )
    return output.model_dump()


class PipelineNodes:
    """Owns the business-logic components used by each pipeline node

    A single instance is reused across invocations of the compiled graph so
    the analyzers aren't reconstructed on every call.
    """

    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.task_classifier = TaskClassifier()
        self.objective_def = ObjectiveDefinition()
        self.criteria_gen = CriteriaGenerator()
        self.heuristic_checker = HeuristicChecker()
        self.evaluator_panel = EvaluatorPanel()
        self.accuracy_calc = AccuracyCalculator()
        self.hitl = HumanInTheLoop()
        self.final_score_calc = FinalScoreCalculator()

    def context_analysis_node(self, state: PipelineState) -> Dict[str, Any]:
        """Node 1: Análise de Contexto"""
        conversation = state["conversation"]
        response, query = _extract_response_and_query(conversation)

        context = self.context_analyzer.analyze(conversation)

        logger.debug("Pipeline node 'context_analysis' complete")
        return {
            "context_analysis": context,
            "response": response,
            "query": query,
        }

    def rule_generation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Node 2: Gerador de Regras"""
        context = state["context_analysis"]
        response = state.get("response", "")
        query = state.get("query", "")

        classification = self.task_classifier.classify(context)
        task_type = classification["task_type"]
        objectives = self.objective_def.get_objectives(task_type)
        criteria = self.criteria_gen.generate(context, objectives)
        heuristics = self.heuristic_checker.check_all(response, query)

        logger.debug("Pipeline node 'rule_generation' complete")
        return {
            "rule_generation": {
                "task_type": task_type,
                "classification": classification,
                "objectives": objectives,
                "criteria": criteria,
                "heuristics": heuristics,
            }
        }

    def jury_evaluation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Node 3: LLM as a Jury"""
        response = state.get("response", "")
        context = state["context_analysis"]
        rules = state["rule_generation"]

        jury_result = self.evaluator_panel.evaluate(
            response, rules["criteria"], context
        )

        confidence = jury_result.get("confidence", 0.5)
        human_review_requested = self.hitl.should_request_review(confidence)
        if human_review_requested:
            logger.info(
                "Jury confidence is low; flagging for human-in-the-loop review"
            )
            self.hitl.request_review(
                evaluation_id=state.get("evaluation_id", "eval"),
                response=response,
                criteria=rules["criteria"],
                automated_score=jury_result.get("average_score", 5.0),
            )

        logger.debug("Pipeline node 'jury_evaluation' complete")
        return {
            "jury_evaluation": {
                **jury_result,
                "human_review_requested": human_review_requested,
            }
        }

    def metrics_aggregation_node(self, state: PipelineState) -> Dict[str, Any]:
        """Node 4: Agregação das métricas"""
        context = state["context_analysis"]
        rules = state["rule_generation"]
        jury_result = state["jury_evaluation"]
        heuristics = rules.get("heuristics", {})

        accuracy = self.accuracy_calc.calculate_accuracy(
            jury_result.get("average_score", 5.0),
            rules.get("criteria", {}),
        )

        final_score = self.final_score_calc.calculate_final_score(
            jury_result, heuristics, accuracy, context
        )
        report = self.final_score_calc.generate_report(
            final_score, evaluation_id=state.get("evaluation_id", "")
        )

        evaluation_output = _build_evaluation_output(
            jury_result, heuristics, accuracy, final_score, report
        )

        logger.debug("Pipeline node 'metrics_aggregation' complete")
        return {
            "metrics_aggregation": {
                "accuracy_metrics": accuracy,
                "final_score_breakdown": final_score,
                "evaluation_report": report,
            },
            "evaluation_output": evaluation_output,
        }
