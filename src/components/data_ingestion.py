"""
data_ingestion.py
─────────────────
Responsibilities
  1. Read the raw CSV from Dataset/
  2. Parse and clean datetime columns
  3. Perform a chronological (time-aware) train/test split — NO shuffle
  4. Save raw copy + train + test splits to artifacts/
  5. Return file paths to the next pipeline stage

Column reference (from the Kaggle dataset)
  Resource ID | Service Name | Usage Quantity | Usage Unit | Region/Zone
  CPU Utilization (%) | Memory Utilization (%) | Network Inbound Data (Bytes)
  Network Outbound Data (Bytes) | Usage Start Date | Usage End Date
  Cost per Quantity ($) | Unrounded Cost ($) | Rounded Cost ($) | Total Cost (INR)
"""

import os
import sys
from dataclasses import dataclass, field

import pandas as pd

from src.exception import CustomException
from src.logger import logger


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DataIngestionConfig:
    """All file-path settings for the ingestion stage.

    Every downstream component reads from these paths, so changing a path
    here automatically propagates through the whole pipeline.
    """
    # Source CSV — already downloaded from Kaggle
    raw_dataset_path: str = os.path.join("Dataset", "cloud_billing_data.csv")

    # Artifact destinations
    artifacts_dir: str = "artifacts"
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    ingestion_report_path: str = os.path.join("artifacts", "ingestion_report.json")

    # Split settings
    test_size: float = 0.20          # last 20 % of timeline → test set
    sort_column: str = "Usage Start Date"   # temporal anchor for the split

    # Datetime columns to parse
    datetime_columns: list = field(default_factory=lambda: [
        "Usage Start Date",
        "Usage End Date",
    ])

    # Expected columns (used for early validation)
    expected_columns: list = field(default_factory=lambda: [
        "Resource ID",
        "Service Name",
        "Usage Quantity",
        "Usage Unit",
        "Region/Zone",
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Network Inbound Data (Bytes)",
        "Network Outbound Data (Bytes)",
        "Usage Start Date",
        "Usage End Date",
        "Cost per Quantity ($)",
        "Unrounded Cost ($)",
        "Rounded Cost ($)",
        "Total Cost (INR)",
    ])


# ── Ingestion Class ───────────────────────────────────────────────────────────

class DataIngestion:
    """Loads the raw GCP billing CSV, validates its structure, performs a
    time-series-aware split, and persists all three artefacts to disk.

    Usage
    -----
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    """

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_artifacts_dir(self) -> None:
        """Create the artifacts/ directory if it does not exist."""
        os.makedirs(self.config.artifacts_dir, exist_ok=True)
        logger.info(f"Artifacts directory ensured: {self.config.artifacts_dir}")

    def _load_csv(self) -> pd.DataFrame:
        """Read the raw CSV into a DataFrame.

        Raises
        ------
        CustomException
            If the file is missing or cannot be parsed.
        """
        try:
            path = self.config.raw_dataset_path
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Dataset not found at '{path}'. "
                    "Make sure 'cloud_billing_data.csv' is inside the Dataset/ folder."
                )
            df = pd.read_csv(path)
            logger.info(f"CSV loaded — shape: {df.shape}, path: {path}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check that every expected column is present.

        Raises
        ------
        CustomException
            If one or more columns are missing.
        """
        try:
            missing = [c for c in self.config.expected_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing columns in dataset: {missing}\n"
                    f"Found columns: {list(df.columns)}"
                )
            logger.info("Column validation passed — all 15 expected columns present.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime columns and coerce errors to NaT.

        Rows where the sort column (Usage Start Date) is NaT after parsing
        are dropped because we cannot place them on the timeline.
        """
        try:
            for col in self.config.datetime_columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                nat_count = df[col].isna().sum()
                if nat_count:
                    logger.warning(
                        f"Column '{col}': {nat_count} rows could not be parsed "
                        "as datetime and were set to NaT."
                    )

            # Drop rows where the sort column is unparseable — they cannot be
            # placed on the timeline and would corrupt the chronological split.
            before = len(df)
            df = df.dropna(subset=[self.config.sort_column]).reset_index(drop=True)
            dropped = before - len(df)
            if dropped:
                logger.warning(
                    f"Dropped {dropped} rows with NaT in '{self.config.sort_column}'."
                )

            logger.info("Datetime parsing complete.")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort the DataFrame by Usage Start Date ascending.

        This is mandatory before the time-series split so that the train set
        always contains earlier records and the test set contains later ones.
        """
        try:
            df = df.sort_values(
                by=self.config.sort_column, ascending=True
            ).reset_index(drop=True)
            logger.info(
                f"Data sorted chronologically by '{self.config.sort_column}'. "
                f"Date range: {df[self.config.sort_column].min()} → "
                f"{df[self.config.sort_column].max()}"
            )
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _log_basic_stats(self, df: pd.DataFrame) -> None:
        """Log a summary of the loaded data for quick sanity checking."""
        try:
            logger.info("── Dataset summary ──────────────────────────────")
            logger.info(f"  Total rows     : {len(df):,}")
            logger.info(f"  Total columns  : {len(df.columns)}")
            logger.info(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
            logger.info(f"  Services       : {df['Service Name'].nunique()} unique")
            logger.info(f"  Regions        : {df['Region/Zone'].nunique()} unique")
            logger.info(f"  Date range     : {df[self.config.sort_column].min()} → "
                        f"{df[self.config.sort_column].max()}")
            logger.info(f"  Null counts    :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
            logger.info(
                f"  Total Cost (INR) — min: {df['Total Cost (INR)'].min():,.0f}  "
                f"max: {df['Total Cost (INR)'].max():,.0f}  "
                f"mean: {df['Total Cost (INR)'].mean():,.0f}"
            )
            logger.info("─────────────────────────────────────────────────")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _chronological_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the sorted DataFrame into train and test sets by position.

        Why position-based and not date-based?
        ─────────────────────────────────────
        Using a fixed date cutoff risks very unequal split sizes if the data
        is not uniformly distributed over time.  A positional split guarantees
        exactly (1 - test_size) rows in train and test_size rows in test,
        while still preserving temporal order because the DataFrame is already
        sorted chronologically.

        Args
        ----
        df : pd.DataFrame
            Chronologically sorted DataFrame.

        Returns
        -------
        train_df, test_df : tuple[pd.DataFrame, pd.DataFrame]
        """
        try:
            split_idx = int(len(df) * (1 - self.config.test_size))
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            test_df = df.iloc[split_idx:].reset_index(drop=True)

            logger.info(
                f"Chronological split complete — "
                f"train: {len(train_df):,} rows "
                f"({train_df[self.config.sort_column].min()} → "
                f"{train_df[self.config.sort_column].max()}), "
                f"test: {len(test_df):,} rows "
                f"({test_df[self.config.sort_column].min()} → "
                f"{test_df[self.config.sort_column].max()})"
            )

            # Sanity check: no temporal leakage — latest train date ≤ earliest test date
            latest_train = train_df[self.config.sort_column].max()
            earliest_test = test_df[self.config.sort_column].min()
            if latest_train > earliest_test:
                logger.warning(
                    f"Temporal overlap detected! Latest train date ({latest_train}) "
                    f"is after earliest test date ({earliest_test}). "
                    "This may happen with duplicate timestamps — review your data."
                )

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _save_artifacts(
        self,
        raw_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Persist all three DataFrames to the artifacts directory as CSV."""
        try:
            raw_df.to_csv(self.config.raw_data_path, index=False)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(f"Saved raw data  → {self.config.raw_data_path}")
            logger.info(f"Saved train data → {self.config.train_data_path}")
            logger.info(f"Saved test data  → {self.config.test_data_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _save_ingestion_report(
        self,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Write a JSON report summarising this ingestion run.

        This report is read by downstream stages (e.g. DataValidation) and
        can also be logged as an MLflow artifact.
        """
        import json

        try:
            report = {
                "total_rows": len(df),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "test_size_pct": round(self.config.test_size * 100, 1),
                "columns": list(df.columns),
                "date_range": {
                    "start": str(df[self.config.sort_column].min()),
                    "end": str(df[self.config.sort_column].max()),
                },
                "train_date_range": {
                    "start": str(train_df[self.config.sort_column].min()),
                    "end": str(train_df[self.config.sort_column].max()),
                },
                "test_date_range": {
                    "start": str(test_df[self.config.sort_column].min()),
                    "end": str(test_df[self.config.sort_column].max()),
                },
                "null_counts": df.isnull().sum().to_dict(),
                "unique_services": int(df["Service Name"].nunique()),
                "unique_regions": int(df["Region/Zone"].nunique()),
                "service_names": sorted(df["Service Name"].unique().tolist()),
                "region_names": sorted(df["Region/Zone"].unique().tolist()),
                "target_stats": {
                    "min": float(df["Total Cost (INR)"].min()),
                    "max": float(df["Total Cost (INR)"].max()),
                    "mean": round(float(df["Total Cost (INR)"].mean()), 2),
                    "median": float(df["Total Cost (INR)"].median()),
                },
            }

            with open(self.config.ingestion_report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Ingestion report saved → {self.config.ingestion_report_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── public API ────────────────────────────────────────────────────────────

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """Run the full ingestion pipeline.

        Steps
        ─────
        1. Create artifacts/ directory
        2. Load raw CSV
        3. Validate columns
        4. Parse datetime columns
        5. Sort chronologically
        6. Log basic statistics
        7. Chronological train/test split
        8. Save all artefacts to disk
        9. Save ingestion report JSON

        Returns
        -------
        tuple[str, str]
            (train_data_path, test_data_path) — consumed by the next
            pipeline stage (DataValidation / DataTransformation).

        Raises
        ------
        CustomException
            Wraps any exception with file name and line number for easy
            debugging.
        """
        logger.info("══════════════════════════════════════════════════")
        logger.info("  DATA INGESTION — started")
        logger.info("══════════════════════════════════════════════════")

        try:
            # 1. Prepare output directory
            self._make_artifacts_dir()

            # 2. Load
            df = self._load_csv()

            # 3. Validate structure
            self._validate_columns(df)

            # 4. Parse datetimes
            df = self._parse_datetimes(df)

            # 5. Sort chronologically
            df = self._sort_chronologically(df)

            # 6. Log stats
            self._log_basic_stats(df)

            # 7. Save raw copy (post datetime-parse, pre-split)
            df.to_csv(self.config.raw_data_path, index=False)

            # 8. Split
            train_df, test_df = self._chronological_split(df)

            # 9. Save train + test
            self._save_artifacts(df, train_df, test_df)

            # 10. Write ingestion report
            self._save_ingestion_report(df, train_df, test_df)

            logger.info("  DATA INGESTION — completed successfully")
            logger.info("══════════════════════════════════════════════════")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise CustomException(e, sys) from e


# ── Entry point (run this file directly to test ingestion standalone) ─────────

if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print("\nIngestion complete.")
    print(f"  Train → {train_path}")
    print(f"  Test  → {test_path}")