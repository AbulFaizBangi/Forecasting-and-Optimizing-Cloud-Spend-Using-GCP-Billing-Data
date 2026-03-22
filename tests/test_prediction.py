"""
tests/test_prediction.py
Unit tests for PredictPipeline and CustomData.
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_input():
    return CustomData(
        service_name           = "Cloud Run",
        region_zone            = "us-central1",
        usage_quantity         = 500.0,
        usage_unit             = "Requests",
        cpu_utilization        = 72.5,
        memory_utilization     = 48.3,
        network_inbound_bytes  = 500_000_000,
        network_outbound_bytes = 500_000_000,
        cost_per_quantity      = 4.5,
        usage_start_date       = "2022-06-15 14:00:00",
        usage_end_date         = "2022-06-15 22:00:00",
    )


@pytest.fixture
def pipeline():
    """Load pipeline — skips if artifacts not present (CI without model)."""
    if not os.path.exists("artifacts/model.pkl"):
        pytest.skip("artifacts/model.pkl not found — run training pipeline first")
    if not os.path.exists("artifacts/preprocessor.pkl"):
        pytest.skip("artifacts/preprocessor.pkl not found — run training pipeline first")
    p = PredictPipeline()
    p._load()
    return p


# ── CustomData tests ──────────────────────────────────────────────────────────

class TestCustomData:

    def test_to_dataframe_shape(self, sample_input):
        """to_dataframe returns a single-row DataFrame."""
        df = sample_input.to_dataframe()
        assert df.shape[0] == 1, "Expected exactly 1 row"

    def test_to_dataframe_has_all_feature_columns(self, sample_input):
        """DataFrame contains all 21 expected feature columns."""
        df = sample_input.to_dataframe()
        expected_numeric = [
            "Usage Quantity", "CPU Utilization (%)", "Memory Utilization (%)",
            "Cost per Quantity ($)", "duration_hours", "cost_per_hour",
            "cpu_mem_product", "log_network", "hour", "day_of_week",
            "day_of_month", "month", "quarter", "is_weekend",
            "lag_1d", "lag_7d", "rolling_7d_mean", "rolling_7d_std",
        ]
        expected_cat = ["Service Name", "Usage Unit", "Region/Zone"]
        for col in expected_numeric + expected_cat:
            assert col in df.columns, f"Column '{col}' missing from feature DataFrame"

    def test_duration_hours_computed(self, sample_input):
        """duration_hours is correctly computed from start/end date."""
        df = sample_input.to_dataframe()
        # 14:00 to 22:00 = 8 hours
        assert abs(df["duration_hours"].iloc[0] - 8.0) < 0.01

    def test_log_network_non_negative(self, sample_input):
        """log_network must always be >= 0."""
        df = sample_input.to_dataframe()
        assert df["log_network"].iloc[0] >= 0

    def test_no_nulls_in_numeric_features(self, sample_input):
        """No NaN values in numeric feature columns."""
        df = sample_input.to_dataframe()
        numeric_cols = [
            "Usage Quantity", "CPU Utilization (%)", "Memory Utilization (%)",
            "Cost per Quantity ($)", "duration_hours", "cost_per_hour",
            "cpu_mem_product", "log_network",
        ]
        for col in numeric_cols:
            assert not df[col].isna().any(), f"NaN found in column '{col}'"

    def test_missing_end_date_handled(self):
        """CustomData handles missing end date gracefully."""
        features = CustomData(
            service_name           = "BigQuery",
            region_zone            = "us-east1",
            usage_quantity         = 100.0,
            usage_unit             = "Requests",
            cpu_utilization        = 50.0,
            memory_utilization     = 40.0,
            network_inbound_bytes  = 100_000_000,
            network_outbound_bytes = 100_000_000,
            cost_per_quantity      = 2.0,
            usage_start_date       = "2022-03-10 08:00:00",
            usage_end_date         = "",       # missing
        )
        df = features.to_dataframe()
        assert df.shape[0] == 1
        # duration_hours defaults to 1.0 when end date is missing
        assert df["duration_hours"].iloc[0] == 1.0

    def test_weekend_detection(self):
        """is_weekend = 1 for Saturday/Sunday."""
        # 2022-01-01 is a Saturday
        features = CustomData(
            service_name="Cloud Storage", region_zone="us-central1",
            usage_quantity=50.0, usage_unit="GB",
            cpu_utilization=10.0, memory_utilization=10.0,
            network_inbound_bytes=1_000_000, network_outbound_bytes=1_000_000,
            cost_per_quantity=1.0, usage_start_date="2022-01-01 10:00:00",
        )
        df = features.to_dataframe()
        assert df["is_weekend"].iloc[0] == 1.0, "2022-01-01 (Saturday) should be weekend"


# ── PredictPipeline tests ─────────────────────────────────────────────────────

class TestPredictPipeline:

    def test_predict_returns_positive_float(self, pipeline, sample_input):
        """Prediction must be a positive float (INR cost)."""
        result = pipeline.predict(sample_input)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result > 0, f"Expected positive cost, got {result}"

    def test_predict_batch_length(self, pipeline, sample_input):
        """predict_batch returns same number of results as inputs."""
        batch = [sample_input] * 5
        results = pipeline.predict_batch(batch)
        assert len(results) == 5

    def test_predict_batch_consistent(self, pipeline, sample_input):
        """Same input always produces the same prediction (deterministic)."""
        r1 = pipeline.predict(sample_input)
        r2 = pipeline.predict(sample_input)
        assert r1 == r2, "Predictions are not deterministic"

    def test_model_info_keys(self, pipeline):
        """model_info returns expected keys."""
        info = pipeline.model_info
        assert "model_type"  in info
        assert "preprocessor" in info
        assert "n_features"   in info

    def test_prediction_in_realistic_range(self, pipeline, sample_input):
        """Prediction should be within a realistic INR range (100 to 10M)."""
        result = pipeline.predict(sample_input)
        assert 100 <= result <= 10_000_000, (
            f"Prediction ₹{result:,.0f} is outside realistic range"
        )