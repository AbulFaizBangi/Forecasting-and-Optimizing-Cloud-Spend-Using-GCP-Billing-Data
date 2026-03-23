"""
tests/test_ingestion.py
Unit tests for DataIngestion component.
"""
import os
import sys
import pytest
import pandas as pd
import tempfile
import shutil

# ── ensure src is importable ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.components.data_ingestion import DataIngestion, DataIngestionConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temp directory for test artifacts."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    dataset   = tmp_path / "Dataset"
    dataset.mkdir()
    return tmp_path


@pytest.fixture
def sample_csv(temp_dir):
    """Write a minimal valid billing CSV to the temp Dataset folder."""
    import numpy as np
    from datetime import datetime, timedelta

    rows = []
    base = datetime(2022, 1, 1)
    services = ["Cloud Run", "BigQuery", "Compute Engine"]
    regions  = ["us-central1", "us-east1", "asia-south1"]
    units    = ["Requests", "Hours", "GB"]

    for i in range(120):
        start = base + timedelta(hours=i * 2)
        end   = start + timedelta(hours=4)
        svc   = services[i % 3]
        reg   = regions[i % 3]
        unit  = units[i % 3]
        qty   = round(10.0 + i * 1.5, 2)
        cpq   = round(1.0 + (i % 5) * 0.5, 4)
        cost  = round(qty * cpq, 4)
        rows.append({
            "Resource ID":                  f"resource_{i}",
            "Service Name":                 svc,
            "Usage Quantity":               qty,
            "Usage Unit":                   unit,
            "Region/Zone":                  reg,
            "CPU Utilization (%)":          round(10 + i % 90, 2),
            "Memory Utilization (%)":       round(20 + i % 80, 2),
            "Network Inbound Data (Bytes)": 100_000_000 + i * 1_000_000,
            "Network Outbound Data (Bytes)":100_000_000 + i * 500_000,
            "Usage Start Date":             start.strftime("%Y-%m-%d %H:%M:%S"),
            "Usage End Date":               end.strftime("%Y-%m-%d %H:%M:%S"),
            "Cost per Quantity ($)":        cpq,
            "Unrounded Cost ($)":           cost,
            "Rounded Cost ($)":             int(cost),
            "Total Cost (INR)":             int(cost * 83),
        })

    csv_path = temp_dir / "Dataset" / "cloud_billing_data.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def ingestion_config(temp_dir, sample_csv):
    """DataIngestionConfig pointing at temp directories."""
    return DataIngestionConfig(
        raw_dataset_path    = str(sample_csv),
        artifacts_dir       = str(temp_dir / "artifacts"),
        raw_data_path       = str(temp_dir / "artifacts" / "raw.csv"),
        train_data_path     = str(temp_dir / "artifacts" / "train.csv"),
        test_data_path      = str(temp_dir / "artifacts" / "test.csv"),
        ingestion_report_path = str(temp_dir / "artifacts" / "ingestion_report.json"),
        test_size           = 0.20,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataIngestion:

    def test_ingestion_returns_paths(self, ingestion_config):
        """initiate_data_ingestion returns two valid file paths."""
        ingestion = DataIngestion(ingestion_config)
        train_path, test_path = ingestion.initiate_data_ingestion()
        assert os.path.exists(train_path), "train.csv was not created"
        assert os.path.exists(test_path),  "test.csv was not created"

    def test_train_test_row_counts(self, ingestion_config):
        """80/20 split produces correct row counts."""
        ingestion = DataIngestion(ingestion_config)
        ingestion.initiate_data_ingestion()

        train = pd.read_csv(ingestion_config.train_data_path)
        test  = pd.read_csv(ingestion_config.test_data_path)

        total = len(train) + len(test)
        assert total == 120, f"Expected 120 total rows, got {total}"
        assert len(train) == 96,  f"Expected 96 train rows, got {len(train)}"
        assert len(test)  == 24,  f"Expected 24 test rows, got {len(test)}"

    def test_no_temporal_leakage(self, ingestion_config):
        """Latest train date must be <= earliest test date."""
        ingestion = DataIngestion(ingestion_config)
        ingestion.initiate_data_ingestion()

        train = pd.read_csv(
            ingestion_config.train_data_path,
            parse_dates=["Usage Start Date"]
        )
        test = pd.read_csv(
            ingestion_config.test_data_path,
            parse_dates=["Usage Start Date"]
        )

        latest_train  = train["Usage Start Date"].max()
        earliest_test = test["Usage Start Date"].min()
        assert latest_train <= earliest_test, (
            f"Temporal leakage: train max {latest_train} > test min {earliest_test}"
        )

    def test_all_columns_present(self, ingestion_config):
        """Train CSV must contain all 15 expected columns."""
        ingestion = DataIngestion(ingestion_config)
        ingestion.initiate_data_ingestion()

        train = pd.read_csv(ingestion_config.train_data_path)
        expected = DataIngestionConfig().expected_columns
        for col in expected:
            assert col in train.columns, f"Column '{col}' missing from train.csv"

    def test_ingestion_report_written(self, ingestion_config):
        """ingestion_report.json must be written with correct keys."""
        ingestion = DataIngestion(ingestion_config)
        ingestion.initiate_data_ingestion()

        import json
        assert os.path.exists(ingestion_config.ingestion_report_path)
        report = json.load(open(ingestion_config.ingestion_report_path))

        for key in ["total_rows", "train_rows", "test_rows",
                    "unique_services", "unique_regions", "target_stats"]:
            assert key in report, f"Key '{key}' missing from ingestion report"

    def test_missing_csv_raises(self, ingestion_config):
        """FileNotFoundError wrapped in CustomException when CSV is missing."""
        from src.exception import CustomException
        bad_config = DataIngestionConfig(
            raw_dataset_path = "nonexistent/path/data.csv",
            artifacts_dir    = ingestion_config.artifacts_dir,
            raw_data_path    = ingestion_config.raw_data_path,
            train_data_path  = ingestion_config.train_data_path,
            test_data_path   = ingestion_config.test_data_path,
        )
        with pytest.raises(CustomException):
            DataIngestion(bad_config).initiate_data_ingestion()