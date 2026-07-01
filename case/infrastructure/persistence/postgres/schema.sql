-- src/infrastructure/persistence/postgres/schema.sql
CREATE TABLE IF NOT EXISTS model_versions (
    version_id UUID PRIMARY KEY,
    family_id UUID NOT NULL,
    project_id UUID NOT NULL,
    version_number INTEGER NOT NULL,
    status VARCHAR(32) NOT NULL,
    persona VARCHAR(32) NOT NULL,
    version_hash TEXT NOT NULL,
    parent_version_id UUID,
    completeness VARCHAR(32) NOT NULL,
    interpretability_score REAL,
    validation_warnings TEXT,
    created_at TIMESTAMP NOT NULL,
    created_by TEXT NOT NULL,
    UNIQUE(family_id, version_number)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY,
    model_version_id UUID NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
    code_snapshot JSONB NOT NULL,
    environment_snapshot JSONB NOT NULL,
    dataset_binding JSONB NOT NULL,
    hyperparameter_bundle JSONB NOT NULL,
    metric_bundle JSONB NOT NULL,
    dataset_hash TEXT NOT NULL,
    training_run_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);
CREATE INDEX idx_training_runs_hash ON training_runs(training_run_hash);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id UUID PRIMARY KEY,
    model_version_id UUID NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
    storage_backend VARCHAR(64) NOT NULL,
    key TEXT NOT NULL,
    checksum TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    framework VARCHAR(64),
    artifact_type VARCHAR(64),
    manifest JSONB,
    tier VARCHAR(32) NOT NULL,
    stored_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS explanation_links (
    id UUID PRIMARY KEY,
    model_version_id UUID NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
    explanation_id TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS outbox (
    id UUID PRIMARY KEY,
    aggregate_type TEXT NOT NULL,
    aggregate_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    status VARCHAR(32) NOT NULL,
    retry_count INTEGER DEFAULT 0
);
CREATE INDEX idx_outbox_status ON outbox(status);

-- Idempotency keys table
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    version_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL
);