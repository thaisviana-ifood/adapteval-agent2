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
3. **LLM as a Jury** (`src/jury`) — a panel of jurors, one per configured
   provider (DeepSeek, OpenAI, Typesafe), answers every rubric criterion
   True/False; the panel's verdict per criterion is a majority vote among
   the jurors that answered it. Jurors run in parallel, and low-confidence
   results are flagged for human-in-the-loop review.
4. **Agregação das métricas** (`src/metrics`) — combines the jury pass rate,
   heuristic and accuracy scores plus a complexity adjustment into a final
   weighted score, confidence, and consolidation report.

Per-module documentation (in Portuguese) lives in [`doc/`](doc/README.md).

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
cp .env.example .env
```

`DEEP_SEEK_API_KEY` is the minimum: it powers the pipeline's own LLM calls
and the conversational front-end. The jury panel defaults to three providers
(`JURY_PROVIDERS` in [`src/config.py`](src/config.py)), so a full jury run
also needs `OPEN_AI_API_KEY` and the `TYPESAFE_*` keys — a juror whose
provider is missing drops out of the vote rather than falling back to
another vendor's model. Langfuse keys are optional; without them tracing and
prompt management degrade to local-only behavior.

### PostgreSQL (for the rubric catalog, and optionally for memory)

```bash
docker compose up -d postgres
```

Stage 2 reads its rubric from the `evaluation_criteria` table: which
criteria are mandatory, which apply to a given (or similar) task type, and
the criteria earlier runs invented. Criteria created at runtime are written
back, so the rubric accumulates instead of being reinvented per call — see
[`doc/rule_generator.md`](doc/rule_generator.md). The table is created and
seeded automatically on first use (`USE_POSTGRES_CRITERIA`, on by default);
without a reachable database the same seed rubric is served from memory and
runtime additions are dropped at exit.

```bash
# Inspect the current rubric (JSON, Markdown table, or replayable SQL)
python3 scripts/dump_criteria.py --format markdown --output -
```

Set `USE_POSTGRES_MEMORY=true` in `.env` to also persist evaluation history
and LangGraph checkpoints to Postgres instead of keeping them in memory.

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

# Measure IntentAnalyzer accuracy against the Bitext goldensets (one LLM
# call per utterance -- keep --limit small)
python3 scripts/eval_intent_accuracy.py --limit 300
python3 scripts/eval_intent_accuracy.py --limit 1080 --workers 12

# Convert the Kaggle multi-turn CSV export into the dataset JSON schema
# (both paths optional; defaults to the Multi-Turn CSV in data/raw/)
python3 scripts/convert_multiturn_csv.py [input.csv] [output.json]

# Publish local prompt templates to Langfuse (required for prompts.py edits
# to take effect -- the Langfuse version wins over the local default)
python3 scripts/push_prompts_to_langfuse.py
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
