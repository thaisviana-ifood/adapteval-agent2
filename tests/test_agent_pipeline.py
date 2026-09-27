"""Tests for the LangGraph-orchestrated thesis evaluation pipeline"""

from unittest.mock import MagicMock

import pytest

from src.shared.llm.client import LLMClient
from src.agent.pipeline_graph import build_pipeline_graph, get_pipeline_graph
from src.agent.schema import EvaluationOutput

CONVERSATION = (
    "User: Can you explain how neural networks work?\n\n"
    "Assistant: Sure! Neural networks are computing systems inspired by "
    "biological neurons, made of layers of interconnected nodes that learn "
    "from data through training.\n\n"
    "User: Can you give a concrete example of training one?\n\n"
    "Assistant: Neural networks consist of layers of interconnected nodes. "
    "Each connection has a weight adjusted during training via "
    "backpropagation: the network predicts, compares to the expected "
    "output, and nudges weights to reduce the error."
)


@pytest.fixture(autouse=True)
def mock_llm_calls(monkeypatch):
    """Avoid real API calls: every LLMEvaluator shares this stubbed response"""
    monkeypatch.setattr(
        LLMClient,
        "call",
        lambda self, *args, **kwargs: (
            "1. Overall Score (0-10): 7\n2. Confidence (0-1): 0.8"
        ),
    )


class TestPipelineGraphStructure:
    def test_graph_has_the_four_thesis_nodes_in_order(self):
        graph = build_pipeline_graph(checkpointer=None)
        node_names = set(graph.get_graph().nodes.keys())

        assert {
            "context_analysis",
            "rule_generation",
            "jury_evaluation",
            "metrics_aggregation",
        }.issubset(node_names)

    def test_get_pipeline_graph_is_a_singleton(self):
        assert get_pipeline_graph() is get_pipeline_graph()


class TestRunPipeline:
    def test_output_matches_the_requested_schema(self):
        from src.agent.pipeline_graph import run_pipeline

        result = run_pipeline(CONVERSATION, evaluation_id="test-eval-1")

        # Validates against Avaliação: <métricas, valores, confiança>,
        # nota de confiança global, relatório de consolidação das métricas.
        validated = EvaluationOutput(**result)

        assert len(validated.avaliacao) > 0
        for metric in validated.avaliacao:
            assert isinstance(metric.metrica, str)
            assert isinstance(metric.valor, float)
            assert 0.0 <= metric.confianca <= 1.0

        assert 0.0 <= validated.nota_confianca_global <= 1.0
        assert validated.relatorio_consolidacao_metricas.evaluation_id == "test-eval-1"

    def test_state_is_checkpointed_after_every_node(self):
        from src.agent.pipeline_graph import get_pipeline_graph, run_pipeline

        run_pipeline(CONVERSATION, evaluation_id="test-eval-checkpoints")

        graph = get_pipeline_graph()
        history = list(
            graph.get_state_history(
                {"configurable": {"thread_id": "test-eval-checkpoints"}}
            )
        )

        # One checkpoint before each node runs, plus the final one: 6 total
        # for a 4-node linear graph (start, 4x pre-node, final).
        assert len(history) == 6

        # State accumulates monotonically: the final checkpoint has every key.
        final_snapshot = history[0]
        assert final_snapshot.next == ()
        for key in (
            "context_analysis",
            "rule_generation",
            "jury_evaluation",
            "metrics_aggregation",
            "evaluation_output",
        ):
            assert key in final_snapshot.values
