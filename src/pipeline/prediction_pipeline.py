"""
prediction_pipeline.py
──────────────────────
Loads the trained preprocessor and model from artifacts/ and serves
single-row predictions at inference time.

Two public classes:

  CustomData
    Dataclass that mirrors the raw input fields a user would provide
    (via Flask form or API JSON). Converts to a DataFrame with the
    exact column names the preprocessor expects.

  PredictPipeline
    Loads preprocessor.pkl + model.pkl once on init, then exposes
    predict(features: CustomData) -> float (INR cost forecast).

Usage — programmatic
---------------------
  from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

  features = CustomData(
      service_name="Cloud Run",
      region_zone="us-central1",
      usage_quantity=500.0,
      usage_unit="Requests",
      cpu_utilization=72.5,
      memory_utilization=48.3,
      network_inbound_bytes=500_000_000,
      network_outbound_bytes=500_000_000,
      cost_per_quantity=4.5,
      usage_start_date="2022-06-15 14:00:00",
  )
  pipeline  = PredictPipeline()
  predicted = pipeline.predict(features)
  print(f"Predicted cost: ₹{predicted:,.0f}")

Usage — Flask app (app.py)
--------------------------
  from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
  pipeline = PredictPipeline()   # load once at startup

  @app.route('/predict', methods=['POST'])
  def predict():
      data     = CustomData(**request.form)
      result   = pipeline.predict(data)
      return render_template('results.html', prediction=result)
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger


# ── CustomData ────────────────────────────────────────────────────────────────

@dataclass
class CustomData:
    """
    Mirrors the raw input fields from the Flask form / API request.

    All fields match the original dataset column names so the mapping
    to the preprocessor feature vector is explicit and auditable.

    Parameters
    ----------
    service_name          : GCP service (e.g. "Cloud Run", "BigQuery")
    region_zone           : Region (e.g. "us-central1", "asia-south1")
    usage_quantity        : Numeric usage amount
    usage_unit            : Unit of usage ("GB", "Hours", "Requests")
    cpu_utilization       : CPU utilization percentage (0–100)
    memory_utilization    : Memory utilization percentage (0–100)
    network_inbound_bytes : Network inbound bytes
    network_outbound_bytes: Network outbound bytes
    cost_per_quantity     : Cost per unit in USD
    usage_start_date      : ISO datetime string "YYYY-MM-DD HH:MM:SS"
    usage_end_date        : Optional ISO datetime string (may be null)
    """
    service_name:           str
    region_zone:            str
    usage_quantity:         float
    usage_unit:             str
    cpu_utilization:        float
    memory_utilization:     float
    network_inbound_bytes:  float
    network_outbound_bytes: float
    cost_per_quantity:      float
    usage_start_date:       str
    usage_end_date:         str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a single-row DataFrame with the exact column names
        and derived features the preprocessor was trained on.

        Mirrors the feature engineering in DataTransformation so
        predictions are consistent with training.
        """
        try:
            # ── Parse datetimes ───────────────────────────────────────────
            start_dt = pd.to_datetime(self.usage_start_date, errors="coerce")
            end_dt   = pd.to_datetime(self.usage_end_date,   errors="coerce") \
                       if self.usage_end_date else pd.NaT

            # ── Datetime features ─────────────────────────────────────────
            hour         = float(start_dt.hour)         if pd.notna(start_dt) else 0.0
            day_of_week  = float(start_dt.dayofweek)    if pd.notna(start_dt) else 0.0
            day_of_month = float(start_dt.day)          if pd.notna(start_dt) else 1.0
            month        = float(start_dt.month)        if pd.notna(start_dt) else 1.0
            quarter      = float(start_dt.quarter)      if pd.notna(start_dt) else 1.0
            is_weekend   = float(start_dt.dayofweek >= 5) if pd.notna(start_dt) else 0.0

            if pd.notna(start_dt) and pd.notna(end_dt):
                duration_hours = max(
                    (end_dt - start_dt).total_seconds() / 3600, 0.01
                )
            else:
                duration_hours = 1.0   # safe default — 1 hour

            # ── Cost / utilisation features ───────────────────────────────
            cost_per_hour   = float(self.cost_per_quantity) * float(self.usage_quantity) \
                              / max(duration_hours, 0.01)
            cpu_mem_product = float(self.cpu_utilization) * float(self.memory_utilization)
            network_total   = float(self.network_inbound_bytes) + \
                              float(self.network_outbound_bytes)
            log_network     = float(np.log1p(network_total))

            # ── Lag / rolling features — use 0 at inference ───────────────
            # At real-time inference we don't have historical daily aggregates
            # for the exact service/region/date. Using 0 here is intentional —
            # the model was trained with StandardScaler so 0 maps to the mean.
            # For batch inference, pass pre-computed lags from the daily series.
            lag_1d          = 0.0
            lag_7d          = 0.0
            rolling_7d_mean = 0.0
            rolling_7d_std  = 0.0

            # ── Build row dict in exact feature order ─────────────────────
            row = {
                # Numeric (18)
                "Usage Quantity":               float(self.usage_quantity),
                "CPU Utilization (%)":          float(self.cpu_utilization),
                "Memory Utilization (%)":       float(self.memory_utilization),
                "Cost per Quantity ($)":        float(self.cost_per_quantity),
                "duration_hours":               duration_hours,
                "cost_per_hour":                cost_per_hour,
                "cpu_mem_product":              cpu_mem_product,
                "log_network":                  log_network,
                "hour":                         hour,
                "day_of_week":                  day_of_week,
                "day_of_month":                 day_of_month,
                "month":                        month,
                "quarter":                      quarter,
                "is_weekend":                   is_weekend,
                "lag_1d":                       lag_1d,
                "lag_7d":                       lag_7d,
                "rolling_7d_mean":              rolling_7d_mean,
                "rolling_7d_std":               rolling_7d_std,
                # Categorical (3)
                "Service Name":                 self.service_name,
                "Usage Unit":                   self.usage_unit,
                "Region/Zone":                  self.region_zone,
            }

            return pd.DataFrame([row])

        except Exception as e:
            raise CustomException(e, sys) from e


# ── PredictPipeline ───────────────────────────────────────────────────────────

class PredictPipeline:
    """
    Loads the trained preprocessor and champion model once, then serves
    predictions for any number of CustomData inputs.

    The preprocessor and model are loaded lazily on first predict() call
    so the Flask app can import this class without file-system side effects
    at import time.
    """

    def __init__(
        self,
        preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl"),
        model_path: str        = os.path.join("artifacts", "model.pkl"),
        feature_schema_path: str = os.path.join("artifacts", "feature_schema.json"),
    ):
        self.preprocessor_path   = preprocessor_path
        self.model_path          = model_path
        self.feature_schema_path = feature_schema_path

        self._preprocessor = None
        self._model        = None
        self._schema       = None

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load preprocessor, model, and schema if not already loaded."""
        if self._preprocessor is not None:
            return

        try:
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessor not found at '{self.preprocessor_path}'. "
                    "Run the training pipeline first."
                )
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model not found at '{self.model_path}'. "
                    "Run the training pipeline first."
                )

            self._preprocessor = joblib.load(self.preprocessor_path)
            self._model        = joblib.load(self.model_path)

            if os.path.exists(self.feature_schema_path):
                with open(self.feature_schema_path) as f:
                    self._schema = json.load(f)

            model_name = type(self._model).__name__
            logger.info(
                f"PredictPipeline loaded — "
                f"model: {model_name}, "
                f"preprocessor: {type(self._preprocessor).__name__}"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, features: CustomData) -> float:
        """
        Generate a cost forecast for the given input features.

        Parameters
        ----------
        features : CustomData
            Raw input from the user / API request.

        Returns
        -------
        float
            Predicted Total Cost in INR (original scale, expm1 applied).
        """
        try:
            self._load()

            # Convert input to DataFrame
            input_df = features.to_dataframe()

            # Apply preprocessor (same ColumnTransformer fitted on train)
            X = self._preprocessor.transform(input_df)

            # Predict in log1p space
            y_log = self._model.predict(X)

            # Inverse transform — back to INR
            y_inr = float(np.expm1(y_log[0]))

            # Clip to non-negative (model can occasionally predict tiny negatives)
            y_inr = max(y_inr, 0.0)

            logger.info(
                f"Prediction: ₹{y_inr:,.0f} INR | "
                f"Service: {features.service_name} | "
                f"Region: {features.region_zone}"
            )

            return round(y_inr, 2)

        except Exception as e:
            raise CustomException(e, sys) from e

    def predict_batch(self, features_list: list[CustomData]) -> list[float]:
        """
        Predict for a list of CustomData inputs in a single batch.

        Parameters
        ----------
        features_list : list[CustomData]

        Returns
        -------
        list[float]
            Predicted costs in INR for each input.
        """
        try:
            self._load()

            # Stack all rows into one DataFrame
            df = pd.concat(
                [f.to_dataframe() for f in features_list],
                ignore_index=True
            )

            X     = self._preprocessor.transform(df)
            y_log = self._model.predict(X)
            y_inr = [round(max(float(np.expm1(v)), 0.0), 2) for v in y_log]

            logger.info(f"Batch prediction: {len(y_inr)} rows processed.")
            return y_inr

        except Exception as e:
            raise CustomException(e, sys) from e

    @property
    def model_info(self) -> dict:
        """Return basic info about the loaded model."""
        self._load()
        info = {
            "model_type":   type(self._model).__name__,
            "preprocessor": type(self._preprocessor).__name__,
        }
        if self._schema:
            info["n_features"]      = self._schema.get("n_features")
            info["target_column"]   = self._schema.get("target_column")
            info["target_inverse"]  = self._schema.get("target_inverse")
        return info


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing prediction pipeline with a sample input...\n")

    features = CustomData(
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

    pipeline  = PredictPipeline()

    print(f"Model info: {pipeline.model_info}")
    print()

    predicted = pipeline.predict(features)
    print(f"Input features:")
    print(f"  Service     : {features.service_name}")
    print(f"  Region      : {features.region_zone}")
    print(f"  Quantity    : {features.usage_quantity} {features.usage_unit}")
    print(f"  CPU / Mem   : {features.cpu_utilization}% / {features.memory_utilization}%")
    print()
    print(f"Predicted Total Cost : ₹{predicted:,.0f} INR")

    # Batch test
    batch = [features] * 3
    batch_preds = pipeline.predict_batch(batch)
    print(f"\nBatch predictions (3 identical inputs): {[f'₹{p:,.0f}' for p in batch_preds]}")