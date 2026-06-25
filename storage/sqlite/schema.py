OBSERVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    observation_type TEXT,
    payload TEXT,
    reliability REAL,
    collected_at INTEGER,
    collector_id TEXT
    
);
"""

SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,

    category TEXT NOT NULL,
    signal_type TEXT NOT NULL,

    subject_type TEXT,
    subject_name TEXT,

    value REAL NOT NULL,
    confidence REAL NOT NULL,

    extractor_id TEXT NOT NULL,

    payload TEXT NOT NULL
);
"""

SIGNAL_OBSERVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS signal_observations (
    signal_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    PRIMARY KEY (signal_id, observation_id),
    FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE,
    FOREIGN KEY (observation_id) REFERENCES observations(id) ON DELETE CASCADE
);
"""

SIGNALS_INDEX = [
    "CREATE INDEX IF NOT EXISTS idx_signals_category ON signals(category);",
    "CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);",
    "CREATE INDEX IF NOT EXISTS idx_signals_subject ON signals(subject_name);",
    "CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_signal_observations_signal ON signal_observations(signal_id);",
    "CREATE INDEX IF NOT EXISTS idx_signal_observations_observation ON signal_observations(observation_id);",
]


SCHEMA = [
    OBSERVATIONS_TABLE,
    SIGNALS_TABLE,
    SIGNAL_OBSERVATIONS_TABLE,
    *SIGNALS_INDEX,
]