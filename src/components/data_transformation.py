"""
data_transformation.py
──────────────────────
Responsibilities
  1. Check validation_report.json — halt if validation failed
  2. Parse datetimes and engineer time-based features
  3. Engineer cost and utilization features
  4. Build lag and rolling features on daily aggregated cost
  5. Build a sklearn ColumnTransformer preprocessor pipeline
  6. Fit on train, transform both train and test (no leakage)
  7. Save preprocessor.pkl, train_transformed.npy, test_transformed.npy
  8. Save feature_schema.json — consumed by prediction pipeline

Feature engineering summary
  ── Datetime features ──────────────────────────────────────────
  hour, day_of_week, day_of_month, month, quarter, is_weekend
  duration_hours  = (end - start).total_seconds() / 3600

  ── Cost / utilisation features ────────────────────────────────
  cost_per_hour   = Unrounded Cost ($) / max(duration_hours, 0.01)
  cpu_mem_product = CPU Utilization (%) x Memory Utilization (%)
  log_network     = log1p(Network Inbound + Network Outbound Bytes)

  ── Lag / rolling features (per Service Name + Region/Zone) ────
  lag_1d, lag_7d   — 1-day and 7-day lag of daily mean cost
  rolling_7d_mean  — 7-day rolling mean of daily cost
  rolling_7d_std   — 7-day rolling std  of daily cost

  ── Preprocessor ───────────────────────────────────────────────
  Numeric  : SimpleImputer(median) -> StandardScaler
  Categorical : SimpleImputer(most_frequent) -> OrdinalEncoder
  Target   : log1p(Total Cost (INR)) — inverse with expm1 at inference
"""

import json
import os
import sys
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DataTransformationConfig:
    # Inputs — written by ingestion
    train_data_path: str          = os.path.join("artifacts", "train.csv")
    test_data_path: str           = os.path.join("artifacts", "test.csv")
    validation_report_path: str   = os.path.join("artifacts", "validation_report.json")

    # Outputs
    preprocessor_path: str        = os.path.join("artifacts", "preprocessor.pkl")
    train_array_path: str         = os.path.join("artifacts", "train_transformed.npy")
    test_array_path: str          = os.path.join("artifacts", "test_transformed.npy")
    feature_schema_path: str      = os.path.join("artifacts", "feature_schema.json")

    # Column names
    target_column: str            = "Total Cost (INR)"
    sort_column: str              = "Usage Start Date"
    datetime_columns: list        = field(default_factory=lambda: [
        "Usage Start Date", "Usage End Date"
    ])

    # Categorical columns for OrdinalEncoder
    categorical_columns: list     = field(default_factory=lambda: [
        "Service Name", "Usage Unit", "Region/Zone"
    ])

    # Numeric columns for StandardScaler (engineered + raw)
    numeric_columns: list         = field(default_factory=lambda: [
        "Usage Quantity",
        "CPU Utilization (%)",
        "Memory Utilization (%)",
        "Cost per Quantity ($)",
        "duration_hours",
        "cost_per_hour",
        "cpu_mem_product",
        "log_network",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "is_weekend",
        "lag_1d",
        "lag_7d",
        "rolling_7d_mean",
        "rolling_7d_std",
    ])


# ── DataTransformation Class ──────────────────────────────────────────────────

class DataTransformation:
    """
    Fits the feature engineering + preprocessing pipeline on train data
    and applies it to both train and test — zero leakage guaranteed.

    Usage
    -----
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation()
    """

    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    # ── Guard: check validation passed ───────────────────────────────────────

    def _check_validation_status(self) -> None:
        """Halt the pipeline if data validation reported critical failures."""
        try:
            if not os.path.exists(self.config.validation_report_path):
                logger.warning(
                    "validation_report.json not found — skipping validation gate. "
                    "Run DataValidation first for production use."
                )
                return

            with open(self.config.validation_report_path) as f:
                report = json.load(f)

            if not report.get("validation_passed", True):
                failures = report.get("critical_failure_details", [])
                raise ValueError(
                    "DataValidation reported CRITICAL failures — "
                    "transformation halted.\nFailures:\n" +
                    "\n".join(f"  x {f}" for f in failures)
                )
            logger.info("Validation gate passed — proceeding with transformation.")
        except ValueError:
            raise
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train = pd.read_csv(self.config.train_data_path)
            test  = pd.read_csv(self.config.test_data_path)
            logger.info(
                f"Loaded train ({len(train):,} rows) and "
                f"test ({len(test):,} rows)."
            )
            return train, test
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            for col in self.config.datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _engineer_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar and duration features from Usage Start Date."""
        try:
            col = "Usage Start Date"
            if col not in df.columns:
                raise ValueError(f"'{col}' not found — cannot engineer datetime features.")

            df["hour"]         = df[col].dt.hour.astype(float)
            df["day_of_week"]  = df[col].dt.dayofweek.astype(float)
            df["day_of_month"] = df[col].dt.day.astype(float)
            df["month"]        = df[col].dt.month.astype(float)
            df["quarter"]      = df[col].dt.quarter.astype(float)
            df["is_weekend"]   = (df[col].dt.dayofweek >= 5).astype(float)

            if "Usage End Date" in df.columns:
                delta = df["Usage End Date"] - df[col]
                df["duration_hours"] = (
                    delta.dt.total_seconds() / 3600
                ).clip(lower=0.01)
            else:
                df["duration_hours"] = np.nan

            logger.info("Datetime features engineered.")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _engineer_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive cost-normalised and utilisation interaction features."""
        try:
            df["cost_per_hour"] = (
                df["Unrounded Cost ($)"] /
                df["duration_hours"].fillna(1.0).clip(lower=0.01)
            )
            df["cpu_mem_product"] = (
                df["CPU Utilization (%)"].fillna(0) *
                df["Memory Utilization (%)"].fillna(0)
            )
            network_total = (
                df["Network Inbound Data (Bytes)"].fillna(0) +
                df["Network Outbound Data (Bytes)"].fillna(0)
            )
            df["log_network"] = np.log1p(network_total)

            logger.info("Cost/utilisation features engineered.")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e

    def _engineer_lag_rolling_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute lag and rolling features using daily aggregated cost
        per (Service Name, Region/Zone) group.

        Train daily series is fitted first — test rows look up from
        that same series to guarantee zero temporal leakage.
        """
        try:
            target     = self.config.target_column
            sort_col   = self.config.sort_column
            group_cols = ["Service Name", "Region/Zone"]
            lag_cols   = ["lag_1d", "lag_7d", "rolling_7d_mean", "rolling_7d_std"]

            # Build daily series from train only
            train_copy = train.copy()
            train_copy["_date"] = pd.to_datetime(
                train_copy[sort_col], errors="coerce"
            ).dt.normalize()

            daily = (
                train_copy.dropna(subset=["_date"])
                .groupby(group_cols + ["_date"])[target]
                .mean()
                .reset_index()
                .rename(columns={target: "_daily_cost"})
                .sort_values(group_cols + ["_date"])
            )

            daily["lag_1d"] = daily.groupby(group_cols)["_daily_cost"].shift(1)
            daily["lag_7d"] = daily.groupby(group_cols)["_daily_cost"].shift(7)
            daily["rolling_7d_mean"] = (
                daily.groupby(group_cols)["_daily_cost"]
                .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
            )
            daily["rolling_7d_std"] = (
                daily.groupby(group_cols)["_daily_cost"]
                .transform(lambda x: x.shift(1).rolling(7, min_periods=2).std())
            )

            merge_keys = group_cols + ["_date"]

            # Map to train
            train_copy = train_copy.merge(
                daily[merge_keys + lag_cols], on=merge_keys, how="left"
            )
            for col in lag_cols:
                train[col] = train_copy[col].values

            # Map to test (look up from train daily series only)
            test_copy = test.copy()
            test_copy["_date"] = pd.to_datetime(
                test_copy[sort_col], errors="coerce"
            ).dt.normalize()
            test_copy = test_copy.merge(
                daily[merge_keys + lag_cols], on=merge_keys, how="left"
            )
            for col in lag_cols:
                test[col] = test_copy[col].values

            # Fill NaN lags with per-group mean from train
            group_means = (
                daily.groupby(group_cols)["_daily_cost"]
                .mean().reset_index()
                .rename(columns={"_daily_cost": "_group_mean"})
            )
            for df in [train, test]:
                tmp = df.merge(group_means, on=group_cols, how="left")
                fill_series = pd.Series(tmp["_group_mean"].values, index=df.index)
                for col in lag_cols:
                    df[col] = df[col].fillna(fill_series)

            logger.info("Lag/rolling features engineered: lag_1d, lag_7d, "
                        "rolling_7d_mean, rolling_7d_std.")
            return train, test
        except Exception as e:
            raise CustomException(e, sys) from e

    def _apply_target_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Separate target and apply log1p transform."""
        try:
            y = df[self.config.target_column].values.astype(float)
            y_log = np.log1p(y)
            df = df.drop(columns=[self.config.target_column])
            return df, y_log
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Preprocessor ─────────────────────────────────────────────────────────

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        ColumnTransformer:
          Numeric  : median imputation -> StandardScaler
          Categorical : most_frequent imputation -> OrdinalEncoder
        """
        try:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline,     self.config.numeric_columns),
                    ("cat", categorical_pipeline, self.config.categorical_columns),
                ],
                remainder="drop",
                verbose_feature_names_out=False,
            )

            logger.info(
                f"Preprocessor built — "
                f"{len(self.config.numeric_columns)} numeric, "
                f"{len(self.config.categorical_columns)} categorical."
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Save artefacts ────────────────────────────────────────────────────────

    def _save_preprocessor(self, preprocessor: ColumnTransformer) -> None:
        try:
            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved -> {self.config.preprocessor_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _save_arrays(self, train_arr: np.ndarray, test_arr: np.ndarray) -> None:
        try:
            np.save(self.config.train_array_path, train_arr)
            np.save(self.config.test_array_path,  test_arr)
            logger.info(f"Arrays saved — train: {train_arr.shape}, "
                        f"test: {test_arr.shape}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _save_feature_schema(
        self, preprocessor: ColumnTransformer, train_arr: np.ndarray
    ) -> None:
        """Save feature schema for use by the prediction pipeline."""
        try:
            schema = {
                "numeric_columns":     self.config.numeric_columns,
                "categorical_columns": self.config.categorical_columns,
                "target_column":       self.config.target_column,
                "target_transform":    "log1p",
                "target_inverse":      "expm1",
                "feature_names_in":    (
                    self.config.numeric_columns +
                    self.config.categorical_columns
                ),
                "n_features":          train_arr.shape[1] - 1,
                "preprocessor_path":   self.config.preprocessor_path,
            }
            with open(self.config.feature_schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Feature schema saved -> {self.config.feature_schema_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def initiate_data_transformation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Run the full transformation pipeline.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, str]
            (train_array, test_array, preprocessor_path)
            Last column of each array is log1p(Total Cost INR).
        """
        logger.info("══════════════════════════════════════════════════")
        logger.info("  DATA TRANSFORMATION — started")
        logger.info("══════════════════════════════════════════════════")

        try:
            self._check_validation_status()

            train, test = self._load_data()

            train = self._parse_datetimes(train)
            test  = self._parse_datetimes(test)

            train = self._engineer_datetime_features(train)
            test  = self._engineer_datetime_features(test)

            train = self._engineer_cost_features(train)
            test  = self._engineer_cost_features(test)

            train, test = self._engineer_lag_rolling_features(train, test)

            train, y_train = self._apply_target_transform(train)
            test,  y_test  = self._apply_target_transform(test)

            preprocessor = self._build_preprocessor()
            X_train = preprocessor.fit_transform(train)
            X_test  = preprocessor.transform(test)

            train_arr = np.c_[X_train, y_train]
            test_arr  = np.c_[X_test,  y_test]

            self._save_preprocessor(preprocessor)
            self._save_arrays(train_arr, test_arr)
            self._save_feature_schema(preprocessor, train_arr)

            logger.info(
                f"Transformation complete — "
                f"train: {train_arr.shape}, test: {test_arr.shape}"
            )
            logger.info("  DATA TRANSFORMATION — completed successfully")
            logger.info("══════════════════════════════════════════════════")

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise CustomException(e, sys) from e


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation()

    print(f"\nTransformation complete.")
    print(f"  Train array shape : {train_arr.shape}")
    print(f"  Test array shape  : {test_arr.shape}")
    print(f"  Preprocessor      : {preprocessor_path}")
    print(f"  Features          : {train_arr.shape[1] - 1}  |  Target: log1p(Total Cost INR)")