# Getting Started with adapteval-agent

This guide gets you from a clean checkout to a working evaluation. For the
architecture and per-module details, see [`doc/`](doc/README.md); for every
configuration variable, see [`doc/config.md`](doc/config.md).

## Quick Start

### 1. Setup environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
```

`.env` essentials:

```env
# Minimum to run anything: powers the pipeline's LLM calls and the
# conversational front-end (DeepSeek, via the OpenAI-compatible API).
DEEP_SEEK_API_KEY=your_key_here

# Needed for a full 3-juror panel (see "Jury Evaluation" below).
OPEN_AI_API_KEY=your_key_here
TYPESAFE_API_KEY=your_key_here
LLM_MODEL_TYPESAFE=your_model_here
TYPESAFE_BASE_URL=your_base_url_here

# Optional: tracing and prompt management.
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

There is no Anthropic key in this project — the default model is
`deepseek-v4-pro` and the jury's other seats are OpenAI and Typesafe.

### 2. Basic usage

The pipeline is a LangGraph state graph. Call it directly — this is the
recommended programmatic path, with a guaranteed output schema:

```python
from src.agent.pipeline_graph import run_pipeline

conversation = """
User: Explain neural networks

Assistant: Neural networks are computational models...

User: Can you give an example?

Assistant: Here's a practical example...
"""

result = run_pipeline(conversation)

print(result["relatorio_consolidacao_metricas"]["final_score"])
print(result["nota_confianca_global"])
```

`run_pipeline` is synchronous. The response it evaluates is extracted from
the conversation itself (the last assistant turn, and the user turn before
it as the query) — you don't pass it separately.

Two things worth knowing up front:

- **`evaluation_id` doubles as the LangGraph `thread_id` and the Langfuse
  `session_id`**, so passing your own id lets you inspect or resume a run
  later by either one.
- **The cache key is the conversation text, not the id.** Re-running the
  same transcript returns the cached output without re-running the graph,
  even under a different `evaluation_id`.

To go through the conversational (`deepagents`) front-end instead:

```python
from src.agent import evaluate_conversation

result = evaluate_conversation(conversation)
```

Same result, at the cost of an extra chat call — prefer `run_pipeline` for
programmatic use.

## Project Components

### 1. Context Analysis (`src/context_analysis`)
- **StructuralAnalyzer** — turn patterns, speaker distribution
- **SemanticAnalyzer** — topics via TF-IDF, plus key phrases, summary and
  sentiment via LLM classifiers
- **ComplexityAnalyzer** — readability, vocabulary diversity
- **IntentAnalyzer** — user intents and assistant objectives (LLM-based;
  supports a closed-set `taxonomy=`)

### 2. Rule Generation (`src/rule_generator`)
- Task type classification (rule-based keyword matching, not an LLM call)
- Objective and success-criteria definition from a static table
- **CriteriaGenerator** — the rubric: base + contextual criteria, plus up to
  3 LLM-proposed dynamic criteria. Every criterion is boolean, phrased as a
  yes/no question.
- **HeuristicChecker** — 4 fast string-level checks, no LLM

### 3. Jury Evaluation (`src/jury`)
- One juror per provider in `JURY_PROVIDERS` — DeepSeek, OpenAI and
  Typesafe by default, matching `MIN_EVALUATORS=3`. Jurors run in parallel.
- **Jurors don't give a 0-10 score**: each answers every rubric criterion
  True/False, and the panel decides each criterion by majority vote among
  the jurors that answered it (a tie fails closed).
- Each juror is built with `fallbacks=()`: if its provider is down it drops
  out of the vote instead of answering with another vendor's model.
- **AccuracyCalculator** — compares against manually annotated data
- **HumanInTheLoop** — flags low-confidence evaluations for review

`RulesBasedEvaluator` also lives in this module but is not part of the panel
(not in `JURY_PROVIDERS`, not instantiated).

### 4. Metrics Aggregation (`src/metrics`)
- **FinalScoreCalculator** — the only one the current pipeline uses:
  weighted consolidation of jury (0.5), heuristic (0.2), accuracy (0.2) and
  a complexity-based context adjustment (0.1)
- `JuryComposition` and `ThresholdCalculator` exist but are only referenced
  by the legacy `src/main.py`

### 5. Memory Management (`src/memory_manager`)
- **CacheManager** — in-memory cache with TTL and a size cap
- **HistoryTracker** — evaluation/metrics history, optionally persisted to
  PostgreSQL
- **ErrorDetector** — structural validation of the result dict
- `CalibrationManager` is instantiated but not called by the current
  pipeline

## Running

```bash
# Full pipeline over a dataset (recommended for batch evaluation)
python3 scripts/run_deep_agent.py --dataset data/raw/diana-agent-csat.json

# Single item from that dataset
python3 scripts/run_deep_agent.py --dataset data/raw/diana-agent-csat.json --item-id <id>

# Legacy, non-LangGraph demo agent (fixed sample conversation)
python3 -m src.main

# Tests
pytest
pytest tests/test_context.py -v
```

Results are written incrementally to `data/output/`.

### PostgreSQL

```bash
docker compose up -d postgres
```

The rubric catalog (`evaluation_criteria`) lives here: stage 2 reads from it
which criteria are mandatory and which apply to the task at hand, and writes
back the criteria it creates at runtime. It is created and seeded on first
use (`USE_POSTGRES_CRITERIA=true`, the default). Inspect it with:

```bash
python3 scripts/dump_criteria.py --format markdown --output -
```

Set `USE_POSTGRES_MEMORY=true` to also persist evaluation history and
LangGraph checkpoints instead of keeping them in process memory. If the
connection fails, all of these degrade back to in-memory — the catalog
serves its seed rubric and forgets runtime additions at exit.

## Understanding Output

`run_pipeline` returns an `EvaluationOutput`
([`src/agent/schema.py`](src/agent/schema.py)):

```python
{
    'avaliacao': [
        # One entry per juror per criterion: valor is the 1.0/0.0 verdict
        {'metrica': 'LLM-deepseek.Relevance', 'valor': 1.0, 'confianca': 0.8},
        {'metrica': 'LLM-typesafe.Relevance', 'valor': 0.0, 'confianca': 0.66},
        # The panel's majority verdict per criterion
        {'metrica': 'jury.Relevance', 'valor': 1.0, 'confianca': 0.75},
        # The 4 consolidation components, on a 0-10 scale
        {'metrica': 'final.jury', 'valor': 7.5, 'confianca': 0.72},
        {'metrica': 'final.heuristic', 'valor': 10.0, 'confianca': 1.0},
        {'metrica': 'final.accuracy', 'valor': 5.0, 'confianca': 0.0},
        {'metrica': 'final.context', 'valor': 5.4, 'confianca': 0.63},
    ],
    'nota_confianca_global': 0.72,
    'relatorio_consolidacao_metricas': {
        'evaluation_id': 'eval-...',
        'final_score': 7.3,
        'confidence': 0.72,
        'recommendation': 'Good - Approved with minor notes',
        'summary': {
            'strength': ['Strong jury evaluation'],
            'weakness': ['Weak accuracy performance'],
        },
        'detailed_breakdown': {...},
    },
}
```

`valor` has a dual meaning, by design: 1.0/0.0 for rubric verdicts
(`jury.*`, `<evaluator>.*`) and a 0-10 scale for the `final.*` metrics.

`final.accuracy = 5.0` with `confianca = 0.0` is the expected output when no
annotated dataset is loaded — it's a neutral placeholder, not a measurement.

## Common Tasks

### Add a custom criterion

```python
from src.rule_generator import CriteriaGenerator

cg = CriteriaGenerator()
criteria = cg.generate(context, objectives)
criteria['weighted_criteria'].append({
    'name': 'CustomMetric',
    'description': 'Does the response satisfy your custom check?',
    'criterion_type': 'boolean',
    'weight': 1.0,
})
```

Phrase `description` as a yes/no question — the jurors answer each criterion
True/False, so a criterion that can't be answered that way will be answered
badly (or abstained on).

### Request human review

```python
from src.agent.pipeline_graph import get_pipeline_nodes

get_pipeline_nodes().hitl.request_review(
    evaluation_id="eval_001",
    response=response,
    criteria=criteria,
    automated_score=score,
    reason="Low confidence",
)
```

`jury_evaluation_node` already does this automatically when panel confidence
falls below the threshold. Reviews are kept in memory only, and
`submit_feedback` has no caller yet — closing the human loop is still open
work.

### Load reference data

```python
from src.jury import AccuracyCalculator

acc = AccuracyCalculator()
acc.load_annotated_data(your_annotated_data)  # [{'score': 8.0, ...}, ...]
accuracy = acc.calculate_accuracy(predicted_score, criteria)
```

### Edit a prompt

Prompt text in [`src/shared/llm/prompts.py`](src/shared/llm/prompts.py) is
only the local default. Once a prompt is published to Langfuse, the Langfuse
version wins — editing the file has no effect until you run:

```bash
python3 scripts/push_prompts_to_langfuse.py
```

## Troubleshooting

### API key issues
- `DEEP_SEEK_API_KEY` must be set. Note that `validate_config()` only runs
  on the legacy `python3 -m src.main` path — `run_pipeline` and the scripts
  don't call it, so a missing key surfaces as a failing LLM call, not as a
  startup warning.
- A missing `OPEN_AI_API_KEY`/`TYPESAFE_*` doesn't crash the run — that
  juror just fails and drops out of the vote, leaving a smaller panel.

### Empty or truncated LLM output
This is almost always a token budget problem with reasoning models, not a
prompt problem: hidden reasoning tokens consume the budget before any
visible output is written. Raise `JURY_LLM_MAX_TOKENS` (jurors) or
`CLASSIFIER_LLM_MAX_TOKENS` (semantic classifiers, dynamic criteria); both
default to `8000`, well above `LLM_MAX_TOKENS`. See
[`doc/config.md`](doc/config.md). Classifier failures are silent by design —
look for the prompt name in `classifier_failures`.

### Database connection errors
- Verify Postgres is running (`docker compose up -d postgres`)
- Check `DB_HOST`/`DB_PORT`; `POSTGRES_HOST_PORT` (default `5433`) is the
  host-side port, in case `5432` is already taken locally

### Unexpected cache hits
`CacheManager` keys on the conversation text. Re-running the same transcript
returns the previous result without touching the graph; call
`get_pipeline_nodes().cache.clear()` or vary the conversation to force a
fresh run.

### Performance
- The jury panel is already parallel (one thread per juror)
- `scripts/eval_intent_accuracy.py` takes `--workers` for parallel LLM calls
- `BATCH_SIZE`, `NUM_WORKERS` and `ASYNC_PROCESSING` are read from the
  environment but not consumed by any module in the current pipeline —
  changing them does nothing
- Set `LOG_LEVEL=WARNING` to reduce log I/O

## Resources

- [README.md](README.md) — overview, setup, observability
- [doc/README.md](doc/README.md) — module documentation index
- [doc/config.md](doc/config.md) — every configuration variable
- [tests/](tests/) — usage examples
- `logs/adapteval.log` — rotating log file
