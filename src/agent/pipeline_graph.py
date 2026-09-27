"""LangGraph-orchestrated thesis pipeline: Análise de Contexto -> Gerador de
Regras -> LLM as a Jury -> Agregação das métricas, with per-node state
persisted by a LangGraph checkpointer"""

import uuid
from contextlib import ExitStack, nullcontext
from typing import Any, Dict, List, Optional

from langfuse import propagate_attributes
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, USE_POSTGRES_MEMORY
from src.shared.llm.langfuse_handler import get_langfuse_callback_handler
from src.shared.logger import get_logger
from src.agent.nodes import PipelineNodes
from src.agent.state import PipelineState

logger = get_logger(__name__)

_exit_stack = ExitStack()
_checkpointer: Optional[BaseCheckpointSaver] = None
_graph: Optional[CompiledStateGraph] = None


def _build_postgres_dsn() -> str:
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_checkpointer() -> BaseCheckpointSaver:
    """Return the process-wide LangGraph checkpointer, creating it on first call

    Uses the same PostgreSQL instance as the rest of the memory manager
    (USE_POSTGRES_MEMORY) when available, so node-level pipeline state and
    evaluation history are persisted side by side. Falls back to an
    in-memory checkpointer otherwise.
    """
    global _checkpointer

    if _checkpointer is not None:
        return _checkpointer

    if USE_POSTGRES_MEMORY:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            saver = _exit_stack.enter_context(
                PostgresSaver.from_conn_string(_build_postgres_dsn())
            )
            saver.setup()
            _checkpointer = saver
            logger.info("Pipeline state will be persisted to PostgreSQL")
            return _checkpointer
        except Exception as e:
            logger.warning(
                f"Could not initialize PostgreSQL checkpointer, falling back "
                f"to in-memory checkpointing: {e}"
            )

    _checkpointer = InMemorySaver()
    return _checkpointer


def build_pipeline_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Build and compile the 4-node thesis pipeline graph"""
    nodes = PipelineNodes()

    graph = StateGraph(PipelineState)
    graph.add_node("context_analysis", nodes.context_analysis_node)
    graph.add_node("rule_generation", nodes.rule_generation_node)
    graph.add_node("jury_evaluation", nodes.jury_evaluation_node)
    graph.add_node("metrics_aggregation", nodes.metrics_aggregation_node)

    graph.add_edge(START, "context_analysis")
    graph.add_edge("context_analysis", "rule_generation")
    graph.add_edge("rule_generation", "jury_evaluation")
    graph.add_edge("jury_evaluation", "metrics_aggregation")
    graph.add_edge("metrics_aggregation", END)

    return graph.compile(checkpointer=checkpointer or get_checkpointer())


def get_pipeline_graph() -> CompiledStateGraph:
    """Return the process-wide compiled pipeline graph, building it on first call"""
    global _graph

    if _graph is None:
        _graph = build_pipeline_graph()

    return _graph


def run_pipeline(
    conversation: str,
    evaluation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full thesis pipeline over a multi-turn conversation

    Args:
        conversation: Multi-turn conversation history ("User: ...\\n\\nAssistant: ...")
        evaluation_id: Optional id; also used as the LangGraph thread_id and
            as the Langfuse session_id, so the run's per-node state and its
            Langfuse trace can both be inspected/resumed later.
        user_id: Optional end-user/tenant identifier attached to the
            Langfuse trace (see
            https://langfuse.com/integrations/frameworks/langgraph).
        tags: Optional Langfuse tags for the trace.
        metadata: Optional extra Langfuse trace metadata.

    Returns:
        The final structured evaluation output (avaliação, nota de
        confiança global, relatório de consolidação das métricas).
    """
    evaluation_id = evaluation_id or f"eval-{uuid.uuid4().hex[:12]}"
    graph = get_pipeline_graph()

    config: Dict[str, Any] = {
        "configurable": {"thread_id": evaluation_id},
        "run_name": "thesis-evaluation-pipeline",
    }

    handler = get_langfuse_callback_handler()
    trace_context = nullcontext()
    if handler is not None:
        config["callbacks"] = [handler]
        trace_context = propagate_attributes(
            trace_name="thesis-evaluation-pipeline",
            session_id=evaluation_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
        )

    with trace_context:
        final_state = graph.invoke(
            {"conversation": conversation, "evaluation_id": evaluation_id},
            config=config,
        )

    return final_state["evaluation_output"]
