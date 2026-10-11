# Jury Architecture & Extension Guide

How the jury panel works and how to add a juror to it. Everything described
here lives in [`src/jury/evaluators.py`](src/jury/evaluators.py); the
Portuguese module documentation is in [`doc/jury.md`](doc/jury.md).

## Overview

Stage 3 of the pipeline is a **multi-vendor panel**: one juror per
configured LLM provider, so no vendor's model is the sole grader of a
response. The panel is extensible by configuration — adding or swapping a
provider is an env/config change, not a code change — and by subclassing
`Evaluator` when a juror needs logic that isn't a chat completion.

**Jurors do not assign a 0-10 score.** Each one answers every rubric
criterion `True`/`False`, and the panel resolves each criterion by majority
vote among the jurors that answered it. This is the core property to respect
when extending the panel: a custom juror that returns a numeric score will
be ignored by the vote.

## Core Components

Everything is in one module, `src/jury/evaluators.py`:

| Class | Role |
|---|---|
| `Evaluator` | Abstract base: `evaluate()` + `get_name()` |
| `LLMEvaluator` | Juror over any OpenAI-compatible provider (DeepSeek, OpenAI) |
| `TypesafeEvaluator` | Juror over Typesafe's typed Q&A API |
| `RulesBasedEvaluator` | Legacy score-based evaluator; **not** in the panel |
| `EvaluatorPanel` | Builds the jurors, runs them in parallel, votes |

`src/jury/__init__.py` only exports `Evaluator`, `EvaluatorPanel`,
`AccuracyCalculator` and `HumanInTheLoop` — the concrete evaluator classes
are imported from `src.jury.evaluators` directly.

### EvaluatorPanel

```python
from src.jury import EvaluatorPanel

panel = EvaluatorPanel()  # defaults to config.JURY_PROVIDERS
result = panel.evaluate(response="...", criteria={...}, context={...})
```

`__init__(providers=None)` takes a list of `{name, model, api_key,
base_url}` dicts — one per juror. `name == "typesafe"` selects
`TypesafeEvaluator`; anything else gets `LLMEvaluator`.

**Responsibilities:**
- Instantiates one juror per provider
- Runs all jurors concurrently in a `ThreadPoolExecutor` (one thread each),
  so panel latency is the slowest juror's, not the sum
- Resolves each rubric criterion by majority vote
- Produces `pass_rate`, per-criterion confidence, and a backward-compatible
  `average_score` (`pass_rate * 10`) for the downstream consumers that still
  expect a number

### The Evaluator contract

```python
from typing import Any, Dict
from src.jury import Evaluator

class MyEvaluator(Evaluator):
    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    def get_name(self) -> str:
        return "MyEvaluator"
```

**Input — `criteria`** is the rubric from
[`CriteriaGenerator`](src/rule_generator/criteria.py). The list the panel
reads is `weighted_criteria`:

```python
{
    "weighted_criteria": [
        {
            "name": "Relevance",
            "description": "Is the response relevant to the query?",
            "criterion_type": "boolean",
            "weight": 1.0,
            "normalized_weight": 0.11,
        },
        ...
    ],
    ...
}
```

Every criterion is boolean, and its `description` is phrased as a yes/no
question. `normalized_weight` is available but **not** used by the vote,
which is a plain majority.

**Input — `context`** is the stage-1 analysis with two keys flattened on top
by `jury_evaluation_node`: `task_type` (from stage 2) and
`complexity_score` (lifted out of the nested `complexity` dict). Read those
two from the top level; everything else stays nested.

**Output** — what the panel actually consumes:

```python
{
    "verdicts": {"Relevance": True, "Clarity": False},  # required
    "confidence": 0.85,                                  # 0-1, overall
    "evaluator": "MyEvaluator",                          # vote key
    "evaluator_type": "custom",                          # optional label
    "reasoning": "...",                                  # optional
    "criterion_confidences": {                           # optional
        "Relevance": 0.9,
        "Clarity": 0.4,
    },
}
```

Three things to get right:

1. **Omit a criterion to abstain.** A criterion missing from `verdicts`
   costs no vote in either direction. This is deliberate: a parse failure
   should remove a juror from that criterion, not push a guessed verdict
   into the vote.
2. **`evaluator` is the vote key.** `_vote_on_criterion` indexes votes by
   `evaluation["evaluator"]`, so two jurors returning the same name
   collapse into one vote. Make `get_name()` unique per instance (that's why
   `LLMEvaluator` names itself `LLM-<provider>`).
3. **`criterion_confidences` beats `confidence`.** If you can say how sure
   you are per criterion, publish it — otherwise every criterion from your
   juror carries the same flat number.

## Built-in Jurors

### LLMEvaluator

A chat completion against one provider, parsed back into verdicts.

```python
from src.jury.evaluators import LLMEvaluator

evaluator = LLMEvaluator(
    evaluator_type="deepseek",
    model="deepseek-v4-pro",
    api_key="...",
    base_url="https://api.deepseek.com",
)
```

- Prompt comes from the Langfuse-managed `llm_evaluation` template; it asks
  for one `"<Criterion Name>: True|False"` line per criterion, then
  `Confidence:` and `Reasoning:`.
- `_parse_evaluation` matches each expected criterion name against the
  response lines; unmatched criteria are left out (abstention).
- The client is built with `max_tokens=JURY_LLM_MAX_TOKENS` (default 8000,
  not the pipeline's 2048) because reasoning models spend budget on hidden
  tokens before writing a verdict.
- The client is built with **`fallbacks=()`**. Unlike every other
  `LLMClient` in the project, a juror has no cross-provider fallback: a
  juror answering with another vendor's model would break the panel's one
  real property. A down provider means the juror drops out of the vote.

### TypesafeEvaluator

Not a chat completion — Typesafe takes typed questions and returns
structured answers, so there's nothing to regex-parse.

```python
from src.jury.evaluators import TypesafeEvaluator

evaluator = TypesafeEvaluator(model="...", api_key="...", base_url="...")
```

- One `"noul"` (yes/no) question per criterion, with instructions mirroring
  the `llm_evaluation` prompt's "fully satisfies, no partial credit"
  standard — deliberately, so this juror isn't held to a looser bar just
  for having a different API shape.
- `noul` comes back as a float 0-1, not a literal boolean:
  `verdicts[name] = noul >= 0.5`, and the distance from the decision
  boundary becomes that verdict's confidence (`abs(noul - 0.5) * 2`). Seeing
  `0.83` instead of `"True"` in Langfuse is the native format, not a bug.
- `TypesafeClient` is plain `httpx`, not the `langfuse.openai` wrapper, so
  this class traces itself manually to stay visible next to the other
  jurors.

### RulesBasedEvaluator

Score-based (0-10) string heuristics: response length, error words, average
sentence length. It is **not** in `JURY_PROVIDERS` and not instantiated by
`_init_evaluators` — it predates the boolean rubric and is kept only as
history. Don't model a new juror on it.

## Adding a Juror

### Option 1: another provider (no code)

The panel is driven by `JURY_PROVIDERS`, derived from `PROVIDERS` in
[`src/config.py`](src/config.py). Add an entry there (model, api key, base
url from the environment) and the panel grows a juror. Any
OpenAI-compatible provider works through `LLMEvaluator` as-is; list it in
`OPENAI_COMPATIBLE_PROVIDERS` if it should also be eligible as an
`LLMClient` fallback elsewhere in the pipeline.

Per-call, you can also pass the list directly:

```python
panel = EvaluatorPanel(providers=[
    {"name": "deepseek", "model": "...", "api_key": "...", "base_url": "..."},
    {"name": "openai",   "model": "...", "api_key": "...", "base_url": "..."},
])
```

Keep `MIN_EVALUATORS` (default 3) in mind: an even-sized panel makes ties
more likely, and ties fail closed.

### Option 2: a custom Evaluator subclass

For a juror that isn't a chat completion (another structured API, a local
model, a deterministic check), implement the contract and append it to the
panel. There is no `add_evaluator()` method — the list is `panel.evaluators`:

```python
from src.jury import EvaluatorPanel

panel = EvaluatorPanel()
panel.evaluators.append(MyEvaluator())
```

To use it in the real pipeline, set it on the shared nodes instance:

```python
from src.agent.pipeline_graph import get_pipeline_nodes

get_pipeline_nodes().evaluator_panel = panel
```

**Your `evaluate()` runs on a worker thread.** Don't share mutable state
between calls without a lock, and don't assume jurors execute in sequence.

## Example: a deterministic juror

A juror doesn't have to call a model. Anything that can answer the rubric
yes/no — and honestly abstain on what it can't judge — is a valid panel
member.

```python
import ast
from typing import Any, Dict

from src.jury import Evaluator


class SyntaxValidityEvaluator(Evaluator):
    """Votes only on criteria it can decide mechanically.

    A Python snippet that doesn't parse cannot be correct, and that's
    knowable without an LLM. Every other criterion is abstained on rather
    than guessed, so this juror adds a vote where it has standing and stays
    out of the rest.
    """

    DECIDABLE = {"Correctness", "Technical Depth"}

    def evaluate(
        self,
        response: str,
        criteria: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if context.get("task_type") != "code_generation":
            # Nothing to say about non-code tasks: abstain on everything.
            return {"verdicts": {}, "confidence": 0.0, "evaluator": self.get_name()}

        try:
            ast.parse(self._extract_code(response))
            parses = True
        except SyntaxError:
            parses = False

        names = {c["name"] for c in criteria.get("weighted_criteria", [])}
        verdicts = {
            name: parses for name in names & self.DECIDABLE
        }

        return {
            "verdicts": verdicts,
            # A syntax error is a fact, not an opinion.
            "confidence": 1.0 if not parses else 0.6,
            "reasoning": f"ast.parse {'succeeded' if parses else 'failed'}",
            "evaluator": self.get_name(),
            "evaluator_type": "syntax",
        }

    def get_name(self) -> str:
        return "Syntax"

    def _extract_code(self, response: str) -> str:
        ...
```

## Best Practices

### 1. Vote, don't score
Return `verdicts: {criterion: bool}`. The panel has no path for a numeric
score — `average_score` is derived from the vote's pass rate downstream, not
collected from jurors.

### 2. Abstain instead of guessing
Leave a criterion out of `verdicts` when you can't judge it (parse failure,
out of scope, missing input). A guessed verdict is worse than a missing one:
it's indistinguishable from a real vote.

### 3. Make confidence mean something
Confidence should track actual uncertainty, because it drives
human-in-the-loop: `jury_evaluation_node` flags a run for review when panel
confidence falls below the threshold. Publish `criterion_confidences` when
you can judge per criterion. A juror that always returns `0.9` quietly
disables that safety net.

### 4. Fail loud at construction, quiet at evaluation
Misconfiguration (missing key, empty base url) should fail when the juror is
built, not silently answer with a default. A failure *during* evaluation
should degrade to abstention — an empty `verdicts` — so the panel shrinks
instead of crashing the pipeline.

### 5. Stay one vendor per juror
Don't give a juror a fallback to another vendor's model. The panel's whole
premise is independent providers; a juror that can answer as its neighbour
turns a 3-vendor panel into a 2-vendor panel without saying so.

### 6. Be traceable
`LLMClient` calls are traced to Langfuse automatically. If your juror talks
to an API through anything else, trace it yourself the way
`TypesafeEvaluator` does, or it will be invisible next to the other jurors.

## Testing a Juror

Test the contract, not a score range:

```python
import pytest

from custom_evaluators import SyntaxValidityEvaluator

CRITERIA = {
    "weighted_criteria": [
        {"name": "Correctness", "description": "Is it correct?",
         "criterion_type": "boolean", "weight": 1.0},
        {"name": "Clarity", "description": "Is it clear?",
         "criterion_type": "boolean", "weight": 0.8},
    ]
}
CONTEXT = {"task_type": "code_generation", "complexity_score": 6}


class TestSyntaxValidityEvaluator:

    @pytest.fixture
    def evaluator(self):
        return SyntaxValidityEvaluator()

    def test_returns_boolean_verdicts(self, evaluator):
        result = evaluator.evaluate("def f(): return 1", CRITERIA, CONTEXT)

        assert all(isinstance(v, bool) for v in result["verdicts"].values())
        assert 0 <= result["confidence"] <= 1
        assert result["evaluator"] == "Syntax"

    def test_abstains_on_criteria_it_cannot_judge(self, evaluator):
        result = evaluator.evaluate("def f(): return 1", CRITERIA, CONTEXT)

        # No opinion on Clarity -- and that must be absence, not False.
        assert "Clarity" not in result["verdicts"]

    def test_abstains_entirely_off_task(self, evaluator):
        result = evaluator.evaluate("hello", CRITERIA, {"task_type": "general"})

        assert result["verdicts"] == {}
```

To test the panel's voting without calling any provider, build it with stub
jurors:

```python
from src.jury import EvaluatorPanel


class StubJuror:
    def __init__(self, name, verdicts):
        self._name, self._verdicts = name, verdicts

    def evaluate(self, response, criteria, context):
        return {"verdicts": self._verdicts, "confidence": 0.9,
                "evaluator": self._name}

    def get_name(self):
        return self._name


def test_tie_fails_closed():
    panel = EvaluatorPanel()
    # `__init__` does `providers or JURY_PROVIDERS`, so an empty list does
    # NOT give an empty panel -- it gives the configured one. Replace the
    # list instead. No provider is contacted at construction time.
    panel.evaluators = [
        StubJuror("a", {"Correctness": True}),
        StubJuror("b", {"Correctness": False}),
    ]

    result = panel.evaluate("...", CRITERIA, CONTEXT)

    assert result["verdicts"]["Correctness"]["passed"] is False
    assert result["verdicts"]["Correctness"]["confidence"] == 0.0
```

## Performance Notes

### Concurrency
The panel already runs its jurors in parallel (`ThreadPoolExecutor`, one
thread per juror), and `pool.map` keeps the reported `evaluations` list in
provider order. Adding a juror costs panel latency only if it's slower than
the current slowest one.

### Caching
Caching happens at the pipeline level, not per juror:
`CacheManager` is keyed on the conversation text, so a repeated transcript
skips the graph entirely. Its constructor takes `max_size_mb` and
`ttl_hours` (not bytes and seconds):

```python
from src.memory_manager import CacheManager

cache = CacheManager(max_size_mb=1024, ttl_hours=24)
```

### Batch evaluation
There is no batch API on the panel. For datasets, call the pipeline per
conversation — `scripts/run_deep_agent.py` does this and writes results
incrementally, so a long run doesn't lose finished work.

## Related Documentation

- [doc/jury.md](doc/jury.md) — this stage in detail (Portuguese)
- [doc/rule_generator.md](doc/rule_generator.md) — where the rubric comes from
- [doc/metrics.md](doc/metrics.md) — what happens to the pass rate afterwards
- [doc/config.md](doc/config.md) — `JURY_PROVIDERS`, token budgets, thresholds
- [doc/agent.md](doc/agent.md) — the LangGraph pipeline around the panel
- [GETTING_STARTED.md](GETTING_STARTED.md) — running it
