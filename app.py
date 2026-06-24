from dataclasses import dataclass

import pandas as pd
import numpy as np
from core.profilers.missingness.pandas_profiler import PandasMissingRateProfiler
from storage.sqlite.connection import SQLiteDB
from storage.sqlite.init import SQLiteInitializer
from storage.sqlite.repositories.observation_repo import ObservationRepository

from core.observation.ingestor import ObservationIngestor
from core.observation.collectors.missingness import ColumnMissingnessObservationCollector, RowsMissingnessObservationCollector


@dataclass
class Context :
    dataset: pd.DataFrame
    sample_size: int
# DB setup
db = SQLiteDB("holmes.db")
SQLiteInitializer(db).init()

# Storage layer
obs_store = ObservationRepository(db)

# Ingestion layer
ingestor = ObservationIngestor(obs_store)

profiler = PandasMissingRateProfiler()
# Collector
collector = RowsMissingnessObservationCollector(profiler)
# collector = ColumnMissingnessObservationCollector(profiler)


def run_pipeline(context):
    
    observations = collector.collect(context)
    ingestor.ingest(observations)



healthy_df = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "age": [23, 35, 41, 29, 51],
    "salary": [50000, 72000, 61000, 58000, 90000],
    "country": ["US", "UK", "CA", "US", "FR"],
    "purchased": [True, False, True, False, True]
})

high_missingness_feat = pd.DataFrame({
    "customer_id": [1,2,3,4,5,6,7,8,9,10],

    "age": [21,25,31,28,45,36,40,22,50,60],

    "salary": [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        52000,
        None,
        None
    ],

    "country": [
        "US","US","UK","FR","DE",
        "CA","US","JP","CN","IT"
    ]
})
full_missing_rows = pd.DataFrame({
    "A": [1, None, None, 4, None],
    "B": [2, None, None, 5, None],
    "C": [3, None, None, 6, None],
})

partially_corrupted = pd.DataFrame({
    "A":[1,None,None,4,5],
    "B":[2,None,3,None,6],
    "C":[3,None,None,None,7],
    "D":[4,None,5,None,8],
    "E":[5,None,None,None,9],
})

context = Context(
    partially_corrupted,
    sample_size = 35
)

run_pipeline(
    context
)