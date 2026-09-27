"""Deep agent (deepagents + LangGraph) front-end for the thesis evaluation pipeline

The agent's only job is to receive a multi-turn conversation, run it through
the 4-stage pipeline (src.agent.pipeline_graph), and hand back the pipeline's
structured JSON output untouched. The actual thesis flow -- and its
per-node state persistence -- lives in the LangGraph graph built by
build_pipeline_graph(); this module only adds a conversational, tool-calling
front-end on top of it via the `deepagents` package.

For programmatic use where the exact JSON schema must be guaranteed, prefer
calling `run_pipeline()` directly instead of going through the agent's
chat-style `invoke`, since the pipeline tool below already returns the
authoritative result -- the agent is only asked to relay it.
"""

import json
from typing import Any, Dict, Optional

from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)
from src.shared.logger import get_logger
from src.agent.pipeline_graph import get_checkpointer, run_pipeline

logger = get_logger(__name__)

SYSTEM_PROMPT = """Você é um agente avaliador de conversas multi-turn.

Fluxo obrigatório para CADA solicitação:
1. Chame a ferramenta `run_thesis_pipeline` exatamente uma vez, passando o
   histórico conversacional completo recebido do usuário.
2. Responda ao usuário APENAS com o JSON retornado pela ferramenta, sem
   comentários, explicações ou blocos de código adicionais -- devolva o
   JSON exatamente como recebido.

Nunca tente calcular métricas, notas ou confiança você mesmo: a ferramenta
já executa todo o pipeline (Análise de Contexto, Gerador de Regras,
LLM as a Jury, Agregação das métricas) e retorna o resultado final."""

_agent = None


@tool
def run_thesis_pipeline(conversation: str) -> str:
    """Executa o pipeline de avaliação (Análise de Contexto -> Gerador de
    Regras -> LLM as a Jury -> Agregação das métricas) sobre um histórico
    conversacional multi-turn e retorna o JSON estruturado final: avaliação
    (métricas, valores, confiança), nota de confiança global e relatório de
    consolidação das métricas.

    Args:
        conversation: Histórico conversacional completo, formatado como
            turnos "User: ..." / "Assistant: ...".
    """
    result = run_pipeline(conversation)
    return json.dumps(result, ensure_ascii=False, default=str)


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
    )


def create_evaluation_deep_agent():
    """Build the deepagents-based evaluator agent

    Returns a compiled LangGraph agent whose sole tool runs the thesis
    pipeline; its checkpointer is shared with the pipeline graph so both the
    agent's own state and the pipeline's per-node state are persisted
    consistently (PostgreSQL when USE_POSTGRES_MEMORY is enabled).
    """
    return create_deep_agent(
        model=_build_model(),
        tools=[run_thesis_pipeline],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
        name="thesis-evaluation-deep-agent",
    )


def get_evaluation_deep_agent():
    """Return the process-wide deep agent, building it on first call"""
    global _agent

    if _agent is None:
        _agent = create_evaluation_deep_agent()

    return _agent


def evaluate_conversation(
    conversation: str, thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run the deep agent over a multi-turn conversation and parse its
    (already-JSON) final answer back into a dict.

    For guaranteed-correct output, prefer
    `src.agent.pipeline_graph.run_pipeline` directly; this helper goes
    through the conversational agent and is mainly useful to exercise/demo
    the deepagents front-end end-to-end.
    """
    agent = get_evaluation_deep_agent()
    thread_id = thread_id or "deep-agent-eval"

    result = agent.invoke(
        {"messages": [HumanMessage(content=conversation)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    final_message = result["messages"][-1]
    content = final_message.content
    if isinstance(content, list):
        # Some providers return content as a list of blocks
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning(
            "Deep agent's final message was not valid JSON; returning raw text"
        )
        return {"raw_response": content}
