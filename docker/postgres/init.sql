-- Schema for the Adaptive Jury Agent memory manager (evaluation & metrics history)

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
