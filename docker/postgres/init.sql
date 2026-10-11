-- Schema for the Adaptive Jury Agent: the rubric catalog read by the rule
-- generator, plus the memory manager's evaluation & metrics history.

-- Catalog of every rubric criterion the pipeline can apply. This table is
-- the source of truth for the rubric (src/rule_generator/criteria_store.py):
--   * `mandatory` rows go into every rubric, whatever the task;
--   * the remaining rows are picked by task (`most_common_tasks`) or by a
--     context trigger evaluated in the rule generator;
--   * criteria the LLM invents at runtime are inserted here, so the next
--     evaluation of the same (or a similar) task can reuse them.
-- The last five columns are the reference library imported from
-- data/rubrics/*.csv (one row per published LLM-as-a-judge metric, with the
-- judge prompt it ships with); they are empty for the pipeline's own
-- criteria. The CSV's `metrica` and `o_que_avalia` are not repeated as
-- columns of their own: they ARE this table's `name` and `description`.
-- Seed rows are not written here: they live in DEFAULT_CRITERIA (and the
-- CSV) and are applied by CriteriaRepository.seed_defaults(), so the
-- catalog has a single source of truth.
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    id SERIAL PRIMARY KEY,
    -- The CSV's `metrica` for imported rows.
    name VARCHAR(255) NOT NULL,
    -- The CSV's `o_que_avalia` for imported rows.
    description TEXT NOT NULL,
    -- "boolean": a yes/no question answered by the LLM jury.
    -- "heuristic": a code check in src/rule_generator/heuristics.py, looked
    -- up by `name` (no LLM call).
    -- "rubric": a published metric imported from the CSV library, which
    -- brings its own judge prompt and its own output scale (see
    -- prompt_rubrica / escala_saida) instead of answering yes/no.
    criterion_type VARCHAR(64) NOT NULL DEFAULT 'boolean',
    mandatory BOOLEAN NOT NULL DEFAULT FALSE,
    avg_weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    -- {task_type: how many rubrics for that task included this criterion},
    -- so the generator can offer the criteria a task usually needs, most
    -- common first. Accumulated by this pipeline's own runs -- the CSV's
    -- recommended tasks are editorial, and live in tarefas_indicadas.
    most_common_tasks JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- How many weights avg_weight averages over. Without it a new sample
    -- could only overwrite the average, never update it.
    weight_samples INTEGER NOT NULL DEFAULT 1,
    -- Framework/paper the metric comes from ("DeepEval", "Ragas", ...).
    -- Empty for criteria this pipeline defines itself.
    avaliador VARCHAR(255) NOT NULL DEFAULT '',
    categoria VARCHAR(255) NOT NULL DEFAULT '',
    -- What the judge must be given (input, actual_output, context, ...).
    entradas_necessarias TEXT NOT NULL DEFAULT '',
    resumo_do_prompt_de_avaliacao TEXT NOT NULL DEFAULT '',
    -- The metric's own output scale ("0-1", "1-5", "Pass/Fail", ...).
    escala_saida VARCHAR(255) NOT NULL DEFAULT '',
    exige_referencia VARCHAR(255) NOT NULL DEFAULT '',
    -- Tasks the source recommends the metric for, in its own words.
    tarefas_indicadas TEXT NOT NULL DEFAULT '',
    fonte TEXT NOT NULL DEFAULT '',
    -- The ready-to-use judge prompt shipped with the metric.
    prompt_rubrica TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Catalogs created before the CSV library existed are migrated here: the
-- table above is CREATE ... IF NOT EXISTS, so on an existing database it is
-- a no-op and these ALTERs are what actually add the columns.
ALTER TABLE evaluation_criteria
    ADD COLUMN IF NOT EXISTS avaliador VARCHAR(255) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS categoria VARCHAR(255) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS entradas_necessarias TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS resumo_do_prompt_de_avaliacao TEXT NOT NULL
        DEFAULT '',
    ADD COLUMN IF NOT EXISTS escala_saida VARCHAR(255) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS exige_referencia VARCHAR(255) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS tarefas_indicadas TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS fonte TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS prompt_rubrica TEXT NOT NULL DEFAULT '';

-- Criteria are identified by name across the pipeline (the jury reports one
-- verdict per name), and the LLM proposes names with its own casing, so
-- uniqueness is case-insensitive while the display casing is preserved.
-- criterion_type is part of the key because the two types are different
-- things with their own namespaces: the `relevance` heuristic (a string
-- check) is not the `Relevance` rubric question the jury answers.
-- avaliador is part of it because a metric name only identifies a metric
-- together with its source: the library has a `Relevance` from Azure, one
-- from Bedrock and one from the G-Eval paper, none of which is this
-- pipeline's own `Relevance` (which has no avaliador at all).
DROP INDEX IF EXISTS idx_evaluation_criteria_name_lower;
CREATE UNIQUE INDEX IF NOT EXISTS idx_evaluation_criteria_identity
    ON evaluation_criteria (lower(name), criterion_type, lower(avaliador));
CREATE INDEX IF NOT EXISTS idx_evaluation_criteria_mandatory
    ON evaluation_criteria (mandatory);
CREATE INDEX IF NOT EXISTS idx_evaluation_criteria_tasks
    ON evaluation_criteria USING gin (most_common_tasks);
CREATE INDEX IF NOT EXISTS idx_evaluation_criteria_categoria
    ON evaluation_criteria (categoria);

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
