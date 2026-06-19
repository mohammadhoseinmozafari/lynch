import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.observation.collector import ObservationCollectorMethod
from core.observation.observation import Observation
from core.observation.type import ObservationType
from core.observation.collectors.missingness import (
    ColumnMissingnessObservationCollector,
    RowsMissingnessObservationCollector,
    MissingnessDistributionObservationCollector
)
from data_investigation.profile.missing_values.profiler import (
    ColumnMissingRateProfile,
    MissingRateProfiler,
    MissingnessDistributionProfile,
    RowsMissingRateProfile,
    
)


class TestColumnMissingnessObservationCollector:
    """Test suite for ColumnMissingnessObservationCollector"""
    
    @pytest.fixture
    def mock_profiler(self):
        return Mock(spec=MissingRateProfiler)
    
    @pytest.fixture
    def collector(self, mock_profiler):
        return ColumnMissingnessObservationCollector(mock_profiler)
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, np.nan, 3, 4, 5],
            'col3': [1, 2, 3, 4, 5]
        })
    
    def test_init(self, mock_profiler):
        """Test collector initialization"""
        collector = ColumnMissingnessObservationCollector(mock_profiler)
        assert collector.observation_type == ObservationType.COLUMN_MISSINGNESS
        assert collector.method_name == ObservationCollectorMethod.NULL_RATE
        assert collector._profiler == mock_profiler
    
    def test_collect(self, collector, mock_profiler, sample_dataframe):
        """Test collect method with valid data"""
        # Mock the profiler's profile_columns method
        mock_profile = {
            'col1': ColumnMissingRateProfile(
                column_name='col1',
                missing_rate=0.2,
                missing_count=1,
                non_missing_count=4,
                total_count=5
            ),
            'col2': ColumnMissingRateProfile(
                column_name='col2',
                missing_rate=0.4,
                missing_count=2,
                non_missing_count=3,
                total_count=5
            ),
            'col3': ColumnMissingRateProfile(
                column_name='col3',
                missing_rate=0.0,
                missing_count=0,
                non_missing_count=5,
                total_count=5
            )
        }
        mock_profiler.profile_columns.return_value = mock_profile
        
        # Mock context
        context = Mock()
        context.dataset = sample_dataframe
        
        # Call collect
        observations = collector.collect(context)
        
        # Assertions
        mock_profiler.profile_columns.assert_called_once_with(sample_dataframe)
        assert len(observations) == 3
        assert all(isinstance(obs, Observation) for obs in observations)
        
        # Check first observation
        obs1 = observations[0]
        assert obs1.type == ObservationType.COLUMN_MISSINGNESS
        assert obs1.collector_id == collector.id
        assert obs1.reliability == 1.0
        assert obs1.payload['column_name'] == 'col1'
        assert obs1.payload['missing_rate'] == 0.2
    
    def test_build_observation(self, collector):
        """Test build_observation method"""
        profile = ColumnMissingRateProfile(
            column_name='test_col',
            missing_rate=0.3,
            missing_count=3,
            non_missing_count=7,
            total_count=10
        )
        
        observation = collector.build_observation(profile)
        
        assert isinstance(observation, Observation)
        assert observation.type == ObservationType.COLUMN_MISSINGNESS
        assert observation.collector_id == collector.id
        assert observation.reliability == 1.0
        assert observation.payload == profile.to_dict()


class TestRowsMissingnessObservationCollector:
    """Test suite for RowsMissingnessObservationCollector"""
    
    @pytest.fixture
    def mock_profiler(self):
        return Mock(spec=MissingRateProfiler)
    
    @pytest.fixture
    def collector(self, mock_profiler):
        return RowsMissingnessObservationCollector(mock_profiler)
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'col1': [np.nan, 2, np.nan, 4, np.nan],
            'col2': [np.nan, np.nan, 3, 4, 5]
        })
    
    def test_init(self, mock_profiler):
        """Test collector initialization"""
        collector = RowsMissingnessObservationCollector(mock_profiler)
        assert collector.observation_type == ObservationType.ROWS_MISSINGNESS
        assert collector.method_name == ObservationCollectorMethod.NULL_RATE
        assert collector._profiler == mock_profiler
    
    def test_collect(self, collector, mock_profiler, sample_dataframe):
        """Test collect method"""
        mock_profile = RowsMissingRateProfile(
            total_rows=5,
            full_missing_rows_count=1,
            full_missing_rows_rate=0.2,
            full_missing_rows_sample_indices=[0],
            high_missing_rate_rows_count=2,
            high_missing_rate_rows_rate=0.4,
            high_missing_rate_rows_sample_indices=[0, 2],
            sample_size=10
        )
        mock_profiler.profile_rows.return_value = mock_profile
        
        context = Mock()
        context.dataset = sample_dataframe
        context.sample_size = 10
        
        observations = collector.collect(context)
        
        mock_profiler.profile_rows.assert_called_once_with(sample_dataframe, 10)
        assert len(observations) == 1
        assert isinstance(observations[0], Observation)
        assert observations[0].type == ObservationType.ROWS_MISSINGNESS
        assert observations[0].payload == mock_profile.to_dict()
    
    def test_build_observation(self, collector):
        """Test build_observation method"""
        profile = RowsMissingRateProfile(
            total_rows=100,
            full_missing_rows_count=5,
            full_missing_rows_rate=0.05,
            full_missing_rows_sample_indices=[1, 2, 3],
            high_missing_rate_rows_count=10,
            high_missing_rate_rows_rate=0.1,
            high_missing_rate_rows_sample_indices=[1, 2, 3, 4],
            sample_size=3
        )
        
        observation = collector.build_observation(profile)
        
        assert isinstance(observation, Observation)
        assert observation.type == ObservationType.ROWS_MISSINGNESS
        assert observation.collector_id == collector.id
        assert observation.reliability == 1.0
        assert observation.payload == profile.to_dict()


class TestMissingnessDistributionObservationCollector:
    """Test suite for MissingnessDistributionObservationCollector"""
    
    @pytest.fixture
    def mock_profiler(self):
        return Mock(spec=MissingRateProfiler)
    
    @pytest.fixture
    def collector(self, mock_profiler):
        return MissingnessDistributionObservationCollector(mock_profiler)
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [np.nan, np.nan, 3, 4, 5],
            'col3': [1, 2, 3, 4, 5]
        })
    
    def test_init(self, mock_profiler):
        """Test collector initialization"""
        collector = MissingnessDistributionObservationCollector(mock_profiler)
        assert collector.observation_type == ObservationType.MISSINGNESS_DISTRIBUTION
        assert collector.method_name == ObservationCollectorMethod.NULL_RATE
        assert collector._profiler == mock_profiler
    
    def test_collect(self, collector, mock_profiler, sample_dataframe):
        """Test collect method"""
        mock_profile = MissingnessDistributionProfile(
            mean_missing_rate=0.2,
            median_missing_rate=0.15,
            std_missing_rate=0.1,
            min_missing_rate=0.0,
            max_missing_rate=0.4,
            p90_missing_rate=0.35,
            p95_missing_rate=0.38,
            p99_missing_rate=0.4
        )
        mock_profiler.profile_distribution.return_value = mock_profile
        
        context = Mock()
        context.dataset = sample_dataframe
        
        observations = collector.collect(context)
        
        mock_profiler.profile_distribution.assert_called_once_with(sample_dataframe)
        assert len(observations) == 1
        assert isinstance(observations[0], Observation)
        assert observations[0].type == ObservationType.MISSINGNESS_DISTRIBUTION
        assert observations[0].payload == mock_profile.to_dict()
    
    def test_build_observation(self, collector):
        """Test build_observation method"""
        profile = MissingnessDistributionProfile(
            mean_missing_rate=0.25,
            median_missing_rate=0.2,
            std_missing_rate=0.15,
            min_missing_rate=0.0,
            max_missing_rate=0.5,
            p90_missing_rate=0.4,
            p95_missing_rate=0.45,
            p99_missing_rate=0.49
        )
        
        observation = collector.build_observation(profile)
        
        assert isinstance(observation, Observation)
        assert observation.type == ObservationType.MISSINGNESS_DISTRIBUTION
        assert observation.collector_id == collector.id
        assert observation.reliability == 1.0
        assert observation.payload == profile.to_dict()


