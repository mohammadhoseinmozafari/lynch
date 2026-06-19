from dataclasses import dataclass, fields
from typing import Any, Dict, List
import pandas as pd

@dataclass
class ColumnMissingRateProfile:
    column_name : str
    missing_rate : float
    missing_count : int
    non_missing_count : int
    total_count : int
    

    def to_dict(self) -> Dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

@dataclass
class RowsMissingRateProfile:
    
    total_rows : int

    full_missing_rows_count : int
    full_missing_rows_rate : float
    full_missing_rows_sample_indices : List[int]

    high_missing_rate_rows_count : int
    high_missing_rate_rows_rate : float
    high_missing_rate_rows_sample_indices : List[int]

    def to_dict(self) -> Dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class MissingnessDistributionProfile:
    mean_missing_rate : float
    median_missing_rate : float
    std_missing_rate : float

    min_missing_rate : float
    max_missing_rate : float

    p90_missing_rate : float
    p95_missing_rate : float
    p99_missing_rate : float

    def to_dict(self) -> Dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


class MissingRateProfiler:

   
    
    def profile_columns (self, df : pd.DataFrame) -> Dict[str, ColumnMissingRateProfile]:
        
        total_rows = len(df)
        missing_counts = df.isna().sum()

        profile = {}

        for column in df.columns:

            missing_count = int(missing_counts[column])
            non_missing_count = total_rows- missing_count

            missing_rate = missing_count/ total_rows if total_rows > 0 else 0.0

            profile[column] = ColumnMissingRateProfile(
                column_name=column,
                missing_count=missing_count,
                non_missing_count=non_missing_count,
                total_count=total_rows,
                missing_rate=missing_rate
            )
            
        
        return profile
    
    def profile_rows (self, df: pd.DataFrame , sample_size : int) -> RowsMissingRateProfile:
        rows_missing_rates = df.isna().mean()
        full_missing_rows_profile = self.profile_full_missing_rows(rows_missing_rates, sample_size)
        high_missing_rate_rows_profile = self.profile_high_missing_rate_rows(rows_missing_rates, threshold= 0.8, sample_size= sample_size)

        return RowsMissingRateProfile(
            total_rows=len(df),
            full_missing_rows_count= full_missing_rows_profile["count"],
            full_missing_rows_rate= full_missing_rows_profile["count"]/ len(df) if len(df) > 0 else 0.0,
            full_missing_rows_sample_indices= full_missing_rows_profile["sample_indices"],
            high_missing_rate_rows_count= high_missing_rate_rows_profile["count"],
            high_missing_rate_rows_rate= high_missing_rate_rows_profile["count"]/ len(df) if len(df) > 0 else 0.0,
            high_missing_rate_rows_sample_indices= high_missing_rate_rows_profile["sample_indices"]
        )
    
    def profile_distribution (self, df: pd.DataFrame) -> MissingnessDistributionProfile:

        rows_missing_rates = df.isna().mean(axis=1)
        

        return MissingnessDistributionProfile(
            mean_missing_rate=float(rows_missing_rates.mean()),
            median_missing_rate=float(rows_missing_rates.median()),
            std_missing_rate=float(rows_missing_rates.std()),
            min_missing_rate=float(rows_missing_rates.min()),
            max_missing_rate=float(rows_missing_rates.max()),
            p90_missing_rate=float(rows_missing_rates.quantile(0.90)),
            p95_missing_rate=float(rows_missing_rates.quantile(0.95)),
            p99_missing_rate=float(rows_missing_rates.quantile(0.99)),
        
        )


    def profile_full_missing_rows (self, rows_missing_rates: pd.Series , sample_size : int) -> Dict[str, Any]:

        fully_missing_rows = rows_missing_rates[rows_missing_rates == 1.0]
        fully_missing_count = len(fully_missing_rows)
        sample_indices = fully_missing_rows.sample(
            (min(fully_missing_count, sample_size)),
            random_state= 42
        ).index.tolist()

        return {
            "count" : fully_missing_count,
            "sample_indices" : sample_indices
        }
    
    
    def profile_high_missing_rate_rows (self, rows_missing_rates: pd.Series, threshold: float, sample_size: int) -> Dict[str, Any]:
        
        high_missing_rate_rows = rows_missing_rates[rows_missing_rates >= threshold]
        high_missing_count = len(high_missing_rate_rows)
        sample_indices = high_missing_rate_rows.sample(
            (min(high_missing_count, sample_size)),
            random_state= 42
        ).index.tolist()

        return {
            "count" : high_missing_count,
            "sample_indices" : sample_indices
        }

    