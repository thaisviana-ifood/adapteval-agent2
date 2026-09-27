"""Deep agent implementation of the thesis evaluation pipeline

Análise de Contexto -> Gerador de Regras -> LLM as a Jury -> Agregação das
métricas, as a LangGraph state graph with per-node checkpointing, fronted by
a deepagents-based conversational agent.
"""

from .deep_agent import create_evaluation_deep_agent, evaluate_conversation
from .pipeline_graph import build_pipeline_graph, get_pipeline_graph, run_pipeline
from .schema import EvaluationOutput, MetricaAvaliada, RelatorioConsolidacao

__all__ = [
    "create_evaluation_deep_agent",
    "evaluate_conversation",
    "build_pipeline_graph",
    "get_pipeline_graph",
    "run_pipeline",
    "EvaluationOutput",
    "MetricaAvaliada",
    "RelatorioConsolidacao",
]
