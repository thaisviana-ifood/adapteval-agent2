# Project Structure & Status

Inventory of what exists in the codebase and what state it's in. For how to
run it, see [GETTING_STARTED.md](GETTING_STARTED.md); for module-by-module
documentation, see [`doc/`](doc/README.md).

## Overview

**adapteval-agent** evaluates assistant responses in multi-turn
conversations through four stages — Análise de Contexto -> Gerador de Regras
-> LLM as a Jury -> Agregação das métricas — orchestrated as a LangGraph
state graph with per-node checkpointing, fronted by a `deepagents`
conversational agent, and traced to Langfuse.

The jury is multi-vendor: DeepSeek, OpenAI and Typesafe, one juror per
provider, so no vendor grades its own model. There is no Anthropic
dependency in this project.

## Component Breakdown

### 1. Shared Services Layer (`src/shared/`)
- **LLM Client** (`llm/client.py`) — OpenAI-compatible wrapper (DeepSeek by
  default), with retry/backoff, `Retry-After` handling, and a configurable
  cross-provider fallback chain
- **JSON Classifier** (`llm/classifier.py`) — shared harness for the
  JSON-returning LLM classifiers: parsing, normalization, and fail-open
  error handling that records the prompt name in `classifier_failures`
- **Prompt Manager** (`llm/prompts.py`) — local templates that double as
  Langfuse-managed prompts; the published Langfuse version takes precedence
- **Typesafe Client** (`llm/typesafe_client.py`) — the structured Q&A API
  behind the Typesafe juror (not OpenAI-compatible, so it has its own
  client and its own manual Langfuse tracing)
- **Langfuse wiring** (`llm/langfuse_client.py`, `llm/langfuse_handler.py`)
  — process-wide client and the LangGraph `CallbackHandler`
- **Infrastructure** (`infra/`) — `PostgresConnector`, plus `VectorStore`
  and `InMemoryQueue`/`TaskProcessor` placeholders not wired into the
  pipeline
- **Utilities** (`utils.py`, `emoji.py`, `logger.py`) — conversation
  parsing, text preprocessing, statistics, emoji sentiment, rotating logs

### 2. Context Analysis (`src/context_analysis/`)
- **StructuralAnalyzer** — turn patterns, speaker distribution
- **SemanticAnalyzer** — topics via TF-IDF, plus key phrases, summary and
  sentiment via LLM classifiers
- **ComplexityAnalyzer** — readability, vocabulary diversity
- **IntentAnalyzer** — user intents and assistant objectives, LLM-based,
  with closed-set taxonomy support

### 3. Rule Generator (`src/rule_generator/`)
- **TaskClassifier** — task type detection (rule-based keyword matching)
- **ObjectiveDefinition** — success criteria and failure conditions
- **CriteriaGenerator** — the boolean rubric: base + contextual criteria,
  plus up to 3 LLM-proposed dynamic criteria
- **HeuristicChecker** — fast, LLM-free validation checks

### 4. Jury Evaluation (`src/jury/`)
- **LLMEvaluator** — one juror per OpenAI-compatible provider, built with
  `fallbacks=()` so a down provider drops out of the vote
- **TypesafeEvaluator** — juror backed by Typesafe's typed yes/no questions
- **EvaluatorPanel** — runs the jurors in parallel and resolves each
  criterion by majority vote among the jurors that answered it
- **AccuracyCalculator** — comparison against annotated reference data
- **HumanInTheLoop** — in-memory review queue for low-confidence runs
- `RulesBasedEvaluator` — present but not part of the panel

### 5. Metrics Aggregation (`src/metrics/`)
- **FinalScoreCalculator** — consolidated weighted score with per-component
  breakdown (the only one the LangGraph pipeline uses)
- **JuryComposition**, **ThresholdCalculator** — implemented, but only
  referenced by the legacy `src/main.py`

### 6. Memory Management (`src/memory_manager/`)
- **CacheManager** — in-memory cache with TTL and size cap, keyed on the
  conversation text
- **HistoryTracker** — evaluation/metrics history, optionally Postgres-backed
- **ErrorDetector** — structural validation of the result dict
- **PostgresMemoryStore** — optional persistence (`USE_POSTGRES_MEMORY`)
- **CalibrationManager** — implemented, instantiated, not called by the
  current pipeline

### 7. Pipeline Orchestration (`src/agent/`)
- **`pipeline_graph.py`** — builds the `StateGraph`, owns the checkpointer
  (Postgres or in-memory) and exposes `run_pipeline()`, the main entry point
- **`nodes.py`** — the four node implementations and the final-output
  assembly
- **`state.py`** / **`schema.py`** — `PipelineState` and the Pydantic
  output contract (`EvaluationOutput`)
- **`deep_agent.py`** — single-tool `deepagents` conversational front-end

### 8. Legacy Agent (`src/main.py`)
- **AdaptiveJuryAgent** — the pre-LangGraph implementation of the same four
  stages: no checkpointing, fixed evaluation ids, and it passes the raw
  context to the jury. Kept as a demo/historical path; see
  [doc/legacy_main.md](doc/legacy_main.md).

## File Statistics

```
Python files:  48 in src/, 6 scripts, 6 test modules
Lines of code: ~8,000 in src/

Module breakdown (lines):
├── src/shared/            1,916   (LLM clients, prompts, infra, utils)
├── src/context_analysis/  1,241   (context understanding)
├── src/memory_manager/    1,136   (cache, history, calibration)
├── src/jury/              1,050   (evaluation)
├── src/rule_generator/      788   (rule generation)
├── src/agent/               740   (LangGraph orchestration)
├── src/metrics/             689   (score aggregation)
└── src/ (config, main)      491
```

## Directory Structure

```
adapteval-agent2/
├── src/
│   ├── config.py              ← Configuration (see doc/config.md)
│   ├── main.py                ← Legacy entry point (AdaptiveJuryAgent)
│   ├── agent/                 ← LangGraph pipeline + deep agent front-end
│   ├── shared/
│   │   ├── llm/               ← Clients, prompts, classifier, Langfuse
│   │   ├── infra/             ← Database & queues
│   │   ├── logger.py
│   │   ├── emoji.py
│   │   └── utils.py
│   ├── context_analysis/      ← Stage 1
│   ├── rule_generator/        ← Stage 2
│   ├── jury/                  ← Stage 3
│   ├── metrics/               ← Stage 4
│   └── memory_manager/        ← Caching, history, calibration
├── scripts/                   ← Batch runs, analysis, goldenset eval,
│                                dataset conversion, prompt publishing
├── tests/                     ← test_shared, test_context, test_memory,
│                                test_llm_client, test_agent_pipeline
├── doc/                       ← Module documentation + architecture diagram
├── data/
│   ├── raw/                   ← Input datasets
│   ├── annotated/             ← Reference data
│   └── output/                ← Generated results
├── docker/postgres/init.sql   ← Schema applied by ensure_schema()
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md
├── GETTING_STARTED.md
└── PROJECT_SETUP.md (this file)
```

## Architecture Highlights

### 1. Pipeline design
Four sequential LangGraph nodes, each writing to its own state key, with a
checkpoint snapshot after every node — a failure mid-run doesn't lose the
stages already completed.

### 2. Multi-vendor jury, boolean rubric
Jurors don't produce a 0-10 score. Every rubric criterion is a yes/no
question answered True/False by each juror, resolved by majority vote (ties
fail closed). Jurors run in parallel and have no cross-vendor fallback, so a
provider outage shrinks the panel instead of silently duplicating a vendor.

### 3. Context-aware evaluation
Criteria are generated from the conversation's complexity, intents and
objectives — including LLM-proposed dynamic criteria — rather than from a
fixed rubric.

### 4. Observability
Every LLM call, prompt version and graph invocation is traced to Langfuse,
with `evaluation_id` shared as the LangGraph `thread_id` and the Langfuse
`session_id`. Without credentials, this degrades to local-only behavior.

## Implemented

- ✅ Four-stage pipeline as a LangGraph state graph with per-node
  checkpointing
- ✅ Multi-vendor jury panel (DeepSeek, OpenAI, Typesafe) with parallel
  jurors and per-criterion majority voting
- ✅ PostgreSQL persistence for evaluation history and graph checkpoints
- ✅ Langfuse tracing and prompt management, with local fallbacks
- ✅ LLM retry/backoff with a configurable cross-provider fallback chain
- ✅ LRU-style cache with TTL, history tracking, structural error detection
- ✅ Pydantic output contract (`EvaluationOutput`)
- ✅ Goldenset accuracy harness for the intent classifier
- ✅ Docker Compose setup (app + Postgres)

## Open work

- ⬜ Close the human-in-the-loop cycle — `submit_feedback` has no caller,
  and reviews live only in process memory
- ⬜ Wire up `CalibrationManager` (instantiated, never called)
- ⬜ Wire up `ThresholdCalculator`/`JuryComposition`, or drop them with
  `src/main.py`
- ⬜ Load annotated reference data — without it, `final.accuracy` is a
  neutral 5.0 at zero confidence
- ⬜ Replace the `VectorStore`/queue placeholders in `shared/infra/`, or
  remove them
- ⬜ Decide the fate of `src/main.py`, which duplicates the pipeline without
  its fixes
- ⬜ Evaluation dashboard / API wrapper
