# adapteval-agent

Adaptive LLM-as-a-Jury evaluation pipeline for multi-turn conversations. It
scores an assistant's responses through four stages — Análise de Contexto,
Gerador de Regras, LLM as a Jury, and Agregação das métricas — orchestrated
as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph and
fronted by a [deepagents](https://github.com/langchain-ai/deepagents)
conversational agent. Runs and traces are observed with
[Langfuse](https://langfuse.com).

## Pipeline stages

1. **Análise de Contexto** (`src/context_analysis`) — structural, semantic,
   complexity and intent analysis of the conversation.
2. **Gerador de Regras** (`src/rule_generator`) — classifies the task,
   defines objectives, generates evaluation criteria, and runs heuristic
   checks.
3. **LLM as a Jury** (`src/jury`) — a panel of LLM evaluators scores the
   response against the generated criteria; low-confidence results are
   flagged for human-in-the-loop review.
4. **Agregação das métricas** (`src/metrics`) — combines jury, heuristic and
   accuracy scores into a final score, confidence, and consolidation report.

Each stage is a node in the graph built by
[`src/agent/pipeline_graph.py`](src/agent/pipeline_graph.py)
(`build_pipeline_graph()` / `run_pipeline()`); per-node state is persisted by
a LangGraph checkpointer (PostgreSQL when `USE_POSTGRES_MEMORY=true`, in
memory otherwise). [`src/agent/deep_agent.py`](src/agent/deep_agent.py) wraps
the graph in a single-tool deepagents front-end
(`run_thesis_pipeline`) for conversational, tool-calling access to the same
pipeline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in DEEP_SEEK_API_KEY and, optionally, Langfuse keys
```

### PostgreSQL (optional, for persistent memory)

```bash
docker compose up -d postgres
```

Set `USE_POSTGRES_MEMORY=true` in `.env` to persist evaluation history and
LangGraph checkpoints to Postgres instead of keeping them in memory.

## Running

```bash
# Single demo evaluation (non-LangGraph AdaptiveJuryAgent)
python3 -m src.main

# Run the LangGraph pipeline / deep agent over a dataset
python3 scripts/run_deep_agent.py --dataset data/raw/diana-agent-csat.json
python3 scripts/run_deep_agent.py --dataset data/raw/diana-agent-csat.json --item-id <id>
python3 scripts/run_deep_agent.py --use-agent  # go through the deepagents front-end instead

# Context/intent analysis only
python3 scripts/run_context_analysis.py --dataset data/raw/diana-agent-csat.json
python3 scripts/run_intent_analysis.py --dataset data/raw/diana-agent-csat.json
```

Results are written incrementally to `data/output/`.

## Observability

LLM calls, prompt management and the LangGraph pipeline/agent runs are all
traced to Langfuse when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are set
in `.env` (`LANGFUSE_BASE_URL` defaults to Langfuse Cloud). Every graph
invocation is traced end-to-end following the
[LangGraph integration guide](https://langfuse.com/integrations/frameworks/langgraph):

- [`src/shared/llm/langfuse_handler.py`](src/shared/llm/langfuse_handler.py)
  holds the process-wide `CallbackHandler`, attached via
  `config={"callbacks": [...]}` on every `graph.invoke()` call.
- `run_pipeline()` and `evaluate_conversation()` wrap their invocation in
  `langfuse.propagate_attributes(...)`, setting the trace's `session_id`
  (the same id used as the LangGraph `thread_id`), `user_id`, `tags` and
  `metadata`. The deep agent's nested `run_thesis_pipeline` tool call
  inherits the same session/user automatically, since it runs inside that
  same context.
- `scripts/run_deep_agent.py` derives `user_id`/`tags`/`metadata` per
  dataset item (tenant hash, dataset name, source trace id) so evaluations
  of the same tenant group under one Langfuse user across runs.

Without Langfuse credentials, tracing and prompt management degrade
gracefully to local-only behavior.

## Tests

```bash
pytest
```
