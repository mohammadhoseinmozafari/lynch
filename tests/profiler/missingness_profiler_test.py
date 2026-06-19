import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from data_investigation.profile.missing_values.profiler import (
    ColumnMissingRateProfile,
    MissingRateProfiler,
    MissingnessDistributionProfile,
    RowsMissingRateProfile,
    
)


class TestMissingRateProfiler:
    """Test suite for MissingRateProfiler"""
    
    @pytest.fixture
    def profiler(self):
        return MissingRateProfiler()
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'col1': [1,      2,      np.nan, 4, 5],
            'col2': [np.nan, np.nan, 3,      4, 5],
            'col3': [1,      2,      3,      4, 5]
        })
    
    def test_profile_columns(self, profiler, sample_dataframe):
        """Test profile_columns method"""
        result = profiler.profile_columns(sample_dataframe)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 'col1' in result
        assert 'col2' in result
        assert 'col3' in result
        
        # Check col1 profile
        col1_profile = result['col1']
        assert isinstance(col1_profile, ColumnMissingRateProfile)
        assert col1_profile.column_name == 'col1'
        assert col1_profile.missing_count == 1
        assert col1_profile.non_missing_count == 4
        assert col1_profile.total_count == 5
        assert col1_profile.missing_rate == 0.2
        
        # Check col2 profile
        col2_profile = result['col2']
        assert col2_profile.missing_count == 2
        assert col2_profile.missing_rate == 0.4
    
    def test_profile_columns_empty_dataframe(self, profiler):
        """Test profile_columns with empty DataFrame"""
        df = pd.DataFrame()
        result = profiler.profile_columns(df)
        assert result == {}
    
    def test_profile_rows(self, profiler, sample_dataframe):
        """Test profile_rows method"""
        result = profiler.profile_rows(sample_dataframe, sample_size=2)
        
        assert isinstance(result, RowsMissingRateProfile)
        assert result.total_rows == 5
        assert result.sample_size == 2
        assert result.full_missing_rows_count == 0  # No rows with all columns missing
        assert result.full_missing_rows_rate == 0.0
        assert result.full_missing_rows_sample_indices == []
        
        # Check high missing rate rows (>80%)
        # Row 0: col1 missing, col2 missing, col3 present -> 66.7% missing
        # Row 1: col1 present, col2 missing, col3 present -> 33.3% missing
        # Row 2: col1 missing, col2 present, col3 present -> 33.3% missing
        # Row 3: all present -> 0% missing
        # Row 4: all present -> 0% missing
        # No row has >80% missing, so high_missing_rate_rows should be 0
        assert result.high_missing_rate_rows_count == 0
        assert result.high_missing_rate_rows_rate == 0.0
    
    def test_profile_rows_with_missing_data(self, profiler):
        """Test profile_rows with data that has full missing rows"""
        df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan, 4, 5],
            'col2': [np.nan, np.nan, 3, 4, 5],
            'col3': [np.nan, 2, 3, 4, 5]
        })
        # Row 0: all missing -> full missing
        # Row 1: col1 missing, col2 missing, col3 present -> 66.7% missing
        # Row 2: col1 missing, col2 present, col3 present -> 33.3% missing
        # Row 3: all present
        # Row 4: all present
        
        result = profiler.profile_rows(df, sample_size=2)
        
        assert result.full_missing_rows_count == 1
        assert result.full_missing_rows_rate == 0.2
        assert len(result.full_missing_rows_sample_indices) == 1
        assert result.full_missing_rows_sample_indices[0] == 0
        
        # High missing rate (>80%) should be 1 (row 0)
        assert result.high_missing_rate_rows_count == 1
        assert result.high_missing_rate_rows_rate == 0.2
    
    def test_profile_distribution(self, profiler, sample_dataframe):
        """Test profile_distribution method"""
        result = profiler.profile_distribution(sample_dataframe)
        
        assert isinstance(result, MissingnessDistributionProfile)
        assert result.mean_missing_rate == pytest.approx(0.2)  # (0.2 + 0.4 + 0.0) / 3
        assert result.median_missing_rate == pytest.approx(1/3)  
        assert result.min_missing_rate == 0.0
        assert result.max_missing_rate == pytest.approx(1/3)
        assert result.p90_missing_rate >= 0.33
        assert result.p95_missing_rate >= 0.33
        assert result.p99_missing_rate >= 0.33
    
    def test_profile_full_missing_rows(self, profiler, sample_dataframe):
        """Test profile_full_missing_rows method"""
        rows_missing_rates = pd.Series([0.0, 0.5, 1.0, 0.3, 1.0])
        
        result = profiler.profile_full_missing_rows(rows_missing_rates, sample_size=10)
        
        assert result['count'] == 2
        assert len(result['sample_indices']) == 2
        assert all(idx in [2, 4] for idx in result['sample_indices'])
    
    def test_profile_full_missing_rows_with_sampling(self, profiler, sample_dataframe):
        """Test profile_full_missing_rows with sample size smaller than count"""
        rows_missing_rates = pd.Series([1.0] * 10 + [0.0] * 5)
        
        result = profiler.profile_full_missing_rows(rows_missing_rates, sample_size=3)
        
        assert result['count'] == 10
        assert len(result['sample_indices']) == 3
    
    def test_profile_full_missing_rows_no_missing(self, profiler, sample_dataframe):
        """Test profile_full_missing_rows with no missing rows"""
        rows_missing_rates = pd.Series([0.0] * 5)
        
        result = profiler.profile_full_missing_rows(rows_missing_rates, sample_size=2)
        
        assert result['count'] == 0
        assert result['sample_indices'] == []
    
    def test_profile_high_missing_rate_rows(self, profiler, sample_dataframe):
        """Test profile_high_missing_rate_rows method"""
        rows_missing_rates = pd.Series([0.9, 0.7, 0.95, 0.8, 0.1])
        
        result = profiler.profile_high_missing_rate_rows(
            rows_missing_rates, 
            threshold=0.8, 
            sample_size=10
        )
        
        assert result['count'] == 3  # 0.9 and 0.95
        assert len(result['sample_indices']) == 3
        assert all(idx in [0, 2, 3] for idx in result['sample_indices'])
    
    def test_profile_high_missing_rate_rows_with_sampling(self, profiler, sample_dataframe):
        """Test profile_high_missing_rate_rows with sampling"""
        rows_missing_rates = pd.Series([0.9] * 10 + [0.1] * 5)
        
        result = profiler.profile_high_missing_rate_rows(
            rows_missing_rates, 
            threshold=0.8, 
            sample_size=3
        )
        
        assert result['count'] == 10
        assert len(result['sample_indices']) == 3
    
    def test_edge_cases(self, profiler):
        """Test edge cases for MissingRateProfiler"""
        # Empty DataFrame
        df_empty = pd.DataFrame()
        assert profiler.profile_columns(df_empty) == {}
        
        # Single column DataFrame
        df_single = pd.DataFrame({'col1': [np.nan, 1, 2, np.nan, 3]})
        result = profiler.profile_columns(df_single)
        assert len(result) == 1
        assert result['col1'].missing_count == 2
        assert result['col1'].missing_rate == 0.4
        
        # DataFrame with all NaN
        df_all_nan = pd.DataFrame({'col1': [np.nan, np.nan, np.nan], 'col2': [np.nan, np.nan, np.nan]})
        result = profiler.profile_columns(df_all_nan)
        assert result['col1'].missing_rate == 1.0
        assert result['col2'].missing_rate == 1.0
        
        # Profile rows with all NaN
        result_rows = profiler.profile_rows(df_all_nan, sample_size=2)
        assert result_rows.full_missing_rows_count == 3
        assert result_rows.high_missing_rate_rows_count == 3