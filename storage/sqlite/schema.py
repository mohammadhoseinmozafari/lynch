OBSERVATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    observation_type TEXT,
    payload TEXT
    reliability REAL,
    collected_at INTEGER,
    collector_id TEXT
    
);
"""

SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    timestamp INTEGER,

    signal_type TEXT,
    subject_type TEXT,
    subject_name TEXT,

    value REAL,
    confidence REAL,

    source_observation_ids TEXT,
    metadata TEXT
);
"""

INVESTIGATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS investigations (
    id TEXT PRIMARY KEY,
    timestamp INTEGER,

    scope TEXT,
    status TEXT,

    root_cause TEXT,
    summary TEXT
);
"""


SCHEMA = [
    OBSERVATIONS_TABLE,
    SIGNALS_TABLE,
    INVESTIGATIONS_TABLE,
]