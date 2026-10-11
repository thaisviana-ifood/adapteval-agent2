-- Schema for the Adaptive Jury Agent: the rubric catalog read by the rule
-- generator, plus the memory manager's evaluation & metrics history.

-- Catalog of every rubric criterion the pipeline can apply. This table is
-- the source of truth for the rubric (src/rule_generator/criteria_store.py):
--   * `mandatory` rows go into every rubric, whatever the task;
--   * the remaining rows are picked by task (`most_common_tasks`) or by a
--     context trigger evaluated in the rule generator;
--   * criteria the LLM invents at runtime are inserted here, so the next
--     evaluation of the same (or a similar) task can reuse them.
-- Seed rows are not written here: they live in DEFAULT_CRITERIA in
-- criteria_store.py and are applied by CriteriaRepository.seed_defaults(),
-- so the catalog has a single source of truth.
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    -- "boolean": a yes/no question answered by the LLM jury.
    -- "heuristic": a code check in src/rule_generator/heuristics.py, looked
    -- up by `name` (no LLM call).
    criterion_type VARCHAR(64) NOT NULL DEFAULT 'boolean',
    mandatory BOOLEAN NOT NULL DEFAULT FALSE,
    avg_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    -- {task_type: how many rubrics for that task included this criterion},
    -- so the generator can offer the criteria a task usually needs, most
    -- common first.
    most_common_tasks JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- How many weights avg_weight averages over. Without it a new sample
    -- could only overwrite the average, never update it.
    weight_samples INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Criteria are identified by name across the pipeline (the jury reports one
-- verdict per name), and the LLM proposes names with its own casing, so
-- uniqueness is case-insensitive while the display casing is preserved.
-- criterion_type is part of the key because the two types are different
-- things with their own namespaces: the `relevance` heuristic (a string
-- check) is not the `Relevance` rubric question the jury answers.
CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluation_criteria_name_lower
    ON evaluation_criteria (lower(name), criterion_type);
CREATE INDEX IF NOT EXISTS idx_evaluation_criteria_mandatory
    ON evaluation_criteria (mandatory);
CREATE INDEX IF NOT EXISTS idx_evaluation_criteria_tasks
    ON evaluation_criteria USING gin (most_common_tasks);

CREATE TABLE IF NOT EXISTS evaluation_history (
    id SERIAL PRIMARY KEY,
    evaluation_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(255) NOT NULL,
    final_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    components JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_evaluation_history_task_type
    ON evaluation_history (task_type);
CREATE INDEX IF NOT EXISTS idx_evaluation_history_created_at
    ON evaluation_history (created_at);

CREATE TABLE IF NOT EXISTS metrics_history (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_metrics_history_metric_name
    ON metrics_history (metric_name);
