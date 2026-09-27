"""LangGraph state schema for the thesis evaluation pipeline"""

from typing import Any, Dict, TypedDict


class PipelineState(TypedDict, total=False):
    """State threaded through the 4 pipeline nodes and persisted by the checkpointer

    Each node only returns the keys it owns; LangGraph merges them into this
    shared state and the checkpointer snapshots it after every node.
    """

    # Input
    conversation: str
    evaluation_id: str

    # Derived from the conversation by the context_analysis node
    response: str
    query: str

    # One key per pipeline stage, in execution order
    context_analysis: Dict[str, Any]
    rule_generation: Dict[str, Any]
    jury_evaluation: Dict[str, Any]
    metrics_aggregation: Dict[str, Any]

    # Final structured output (built by the metrics_aggregation node)
    evaluation_output: Dict[str, Any]
