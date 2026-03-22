"""
training_pipeline.py
────────────────────
Wires the four components into a single callable end-to-end pipeline:

  DataIngestion
      ↓  artifacts/train.csv, test.csv
  DataValidation
      ↓  artifacts/validation_report.json
  DataTransformation
      ↓  artifacts/train_transformed.npy, test_transformed.npy, preprocessor.pkl
  ModelTrainer
      ↓  artifacts/model.pkl, model_report.json, mlruns/

Each stage reads its inputs from artifacts/ and writes outputs back to
artifacts/ — no in-memory chaining, so any stage can be re-run in isolation.

Usage
-----
  # From project root:
  PYTHONPATH=. python src/pipeline/training_pipeline.py

  # Or import and call programmatically:
  from src.pipeline.training_pipeline import TrainingPipeline
  pipeline = TrainingPipeline()
  pipeline.run()
"""

import json
import sys
import time
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.data_validation import DataValidation, DataValidationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logger


# ── Pipeline Config ───────────────────────────────────────────────────────────

@dataclass
class TrainingPipelineConfig:
    """
    Controls which stages run and whether to halt on validation failure.
    Set skip_* flags to True to resume from a specific stage without
    re-running earlier ones (useful during development).
    """
    halt_on_validation_failure: bool = True
    skip_ingestion:     bool = False
    skip_validation:    bool = False
    skip_transformation: bool = False
    skip_training:      bool = False


# ── Training Pipeline ─────────────────────────────────────────────────────────

class TrainingPipeline:
    """
    Orchestrates the full ML training workflow.

    Each stage is wrapped in its own try/except so failures are clearly
    attributed to the correct component in logs.
    """

    def __init__(self, config: TrainingPipelineConfig = TrainingPipelineConfig()):
        self.config = config

    def _log_stage(self, name: str, action: str) -> None:
        border = "─" * 50
        logger.info(border)
        logger.info(f"  STAGE: {name} — {action}")
        logger.info(border)

    # ── Stage 1: Data Ingestion ───────────────────────────────────────────────

    def _run_ingestion(self) -> tuple[str, str]:
        self._log_stage("Data Ingestion", "started")
        try:
            ingestion = DataIngestion(DataIngestionConfig())
            train_path, test_path = ingestion.initiate_data_ingestion()
            self._log_stage("Data Ingestion", "completed")
            return train_path, test_path
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Stage 2: Data Validation ──────────────────────────────────────────────

    def _run_validation(self) -> bool:
        self._log_stage("Data Validation", "started")
        try:
            validator = DataValidation(DataValidationConfig())
            passed, report_path = validator.initiate_data_validation()

            if not passed and self.config.halt_on_validation_failure:
                report = json.load(open(report_path))
                failures = report.get("critical_failure_details", [])
                raise ValueError(
                    "Training pipeline halted — DataValidation reported "
                    f"{len(failures)} critical failure(s):\n" +
                    "\n".join(f"  x {f}" for f in failures)
                )

            if not passed:
                logger.warning(
                    "Validation failures detected but halt_on_validation_failure=False — "
                    "continuing with caution."
                )

            self._log_stage("Data Validation", "completed")
            return passed
        except ValueError:
            raise
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Stage 3: Data Transformation ─────────────────────────────────────────

    def _run_transformation(self) -> tuple:
        self._log_stage("Data Transformation", "started")
        try:
            transformer = DataTransformation(DataTransformationConfig())
            train_arr, test_arr, preprocessor_path = (
                transformer.initiate_data_transformation()
            )
            self._log_stage("Data Transformation", "completed")
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Stage 4: Model Training ───────────────────────────────────────────────

    def _run_training(self) -> tuple[dict, str]:
        self._log_stage("Model Training", "started")
        try:
            trainer = ModelTrainer(ModelTrainerConfig())
            metrics, model_path = trainer.initiate_model_training()
            self._log_stage("Model Training", "completed")
            return metrics, model_path
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute all pipeline stages in order.

        Returns
        -------
        dict
            Final test metrics from the champion model.
        """
        logger.info("══════════════════════════════════════════════════")
        logger.info("  TRAINING PIPELINE — started")
        logger.info("══════════════════════════════════════════════════")

        pipeline_start = time.time()
        stage_times    = {}

        try:
            # ── Stage 1 ───────────────────────────────────────────────────
            if not self.config.skip_ingestion:
                t = time.time()
                self._run_ingestion()
                stage_times["ingestion"] = round(time.time() - t, 2)
            else:
                logger.info("Skipping ingestion (skip_ingestion=True)")

            # ── Stage 2 ───────────────────────────────────────────────────
            if not self.config.skip_validation:
                t = time.time()
                self._run_validation()
                stage_times["validation"] = round(time.time() - t, 2)
            else:
                logger.info("Skipping validation (skip_validation=True)")

            # ── Stage 3 ───────────────────────────────────────────────────
            if not self.config.skip_transformation:
                t = time.time()
                self._run_transformation()
                stage_times["transformation"] = round(time.time() - t, 2)
            else:
                logger.info("Skipping transformation (skip_transformation=True)")

            # ── Stage 4 ───────────────────────────────────────────────────
            if not self.config.skip_training:
                t = time.time()
                metrics, model_path = self._run_training()
                stage_times["training"] = round(time.time() - t, 2)
            else:
                logger.info("Skipping training (skip_training=True)")
                metrics    = {}
                model_path = "artifacts/model.pkl"

            # ── Summary ───────────────────────────────────────────────────
            total_time = round(time.time() - pipeline_start, 2)

            logger.info("══════════════════════════════════════════════════")
            logger.info("  TRAINING PIPELINE — completed successfully")
            logger.info(f"  Total time : {total_time}s")
            for stage, t in stage_times.items():
                logger.info(f"    {stage:<16} : {t}s")
            if metrics:
                logger.info(f"  Champion MAPE  : {metrics.get('test_mape', 'N/A')}%")
                logger.info(f"  Champion R²    : {metrics.get('test_r2', 'N/A')}")
                logger.info(f"  Model path     : {model_path}")
            logger.info("══════════════════════════════════════════════════")

            return metrics

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise CustomException(e, sys) from e


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    metrics  = pipeline.run()

    print("\nPipeline complete.")
    if metrics:
        print(f"  Champion MAPE  : {metrics.get('test_mape', 'N/A')}%")
        print(f"  Champion R²    : {metrics.get('test_r2', 'N/A')}")
        print(f"  RMSE (INR)     : {metrics.get('test_rmse_inr', 'N/A'):,}")