"""
model_trainer.py
────────────────
Responsibilities
  1. Load train/test arrays from artifacts/
  2. Run a Prophet baseline on daily aggregated cost (no feature engineering)
  3. Train XGBoost and LightGBM with TimeSeriesSplit cross-validation
  4. Tune the best model with Optuna (50 trials)
  5. Log every run to MLflow (params, metrics, feature importances)
  6. Register the champion model in the local MLflow Model Registry
  7. Save champion model as artifacts/model.pkl
  8. Save model_report.json with final metrics

Metrics tracked
  RMSE   — root mean squared error (in log1p space and INR space)
  MAE    — mean absolute error
  MAPE   — mean absolute percentage error (on original INR scale)
  R²     — coefficient of determination

Target note
  y is log1p(Total Cost INR) — predictions are expm1'd back to INR
  for MAPE and INR-space RMSE reporting.
"""

import json
import os
import sys
import warnings
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
import xgboost as xgb

from src.exception import CustomException
from src.logger import logger

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ModelTrainerConfig:
    # Inputs
    train_array_path: str       = os.path.join("artifacts", "train_transformed.npy")
    test_array_path: str        = os.path.join("artifacts", "test_transformed.npy")
    feature_schema_path: str    = os.path.join("artifacts", "feature_schema.json")
    train_csv_path: str         = os.path.join("artifacts", "train.csv")

    # Outputs
    model_path: str             = os.path.join("artifacts", "model.pkl")
    model_report_path: str      = os.path.join("artifacts", "model_report.json")

    # MLflow
    mlflow_tracking_uri: str    = "./mlruns"
    mlflow_experiment_name: str = "GCP-Billing-Forecasting"
    registered_model_name: str  = "GCP-Billing-Forecaster"

    # Training
    n_cv_splits: int            = 3      # TimeSeriesSplit folds
    optuna_trials: int          = 40     # hyperparameter search trials
    random_state: int           = 42


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Compute metrics in both log1p space and original INR space.

    Parameters
    ----------
    y_true_log : log1p-transformed ground truth
    y_pred_log : log1p-transformed predictions
    prefix     : string prefix for metric keys (e.g. "train_" or "test_")
    """
    # Log-space metrics
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log  = float(mean_absolute_error(y_true_log, y_pred_log))
    r2       = float(r2_score(y_true_log, y_pred_log))

    # Original INR-space metrics
    y_true_inr = np.expm1(y_true_log)
    y_pred_inr = np.expm1(y_pred_log)

    rmse_inr = float(np.sqrt(mean_squared_error(y_true_inr, y_pred_inr)))
    mae_inr  = float(mean_absolute_error(y_true_inr, y_pred_inr))

    # MAPE — avoid division by zero
    mask = y_true_inr > 0
    mape = float(
        np.mean(np.abs((y_true_inr[mask] - y_pred_inr[mask]) / y_true_inr[mask])) * 100
    ) if mask.sum() > 0 else float("nan")

    return {
        f"{prefix}rmse_log":  round(rmse_log, 6),
        f"{prefix}mae_log":   round(mae_log,  6),
        f"{prefix}r2":        round(r2,        6),
        f"{prefix}rmse_inr":  round(rmse_inr,  2),
        f"{prefix}mae_inr":   round(mae_inr,   2),
        f"{prefix}mape":      round(mape,       4),
    }


# ── ModelTrainer Class ────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Trains XGBoost and LightGBM forecasting models with Optuna tuning
    and MLflow experiment tracking.

    Usage
    -----
    trainer = ModelTrainer()
    metrics, model_path = trainer.initiate_model_training()
    """

    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        self.feature_names: list = []

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_mlflow(self) -> None:
        try:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            logger.info(
                f"MLflow tracking URI: {self.config.mlflow_tracking_uri} | "
                f"Experiment: {self.config.mlflow_experiment_name}"
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            train = np.load(self.config.train_array_path)
            test  = np.load(self.config.test_array_path)

            X_train, y_train = train[:, :-1], train[:, -1]
            X_test,  y_test  = test[:, :-1],  test[:, -1]

            schema = json.load(open(self.config.feature_schema_path))
            self.feature_names = schema.get("feature_names_in", [
                f"f{i}" for i in range(X_train.shape[1])
            ])

            logger.info(
                f"Arrays loaded — X_train: {X_train.shape}, "
                f"X_test: {X_test.shape}, features: {len(self.feature_names)}"
            )
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Cross-validated evaluation ────────────────────────────────────────────

    def _cv_score(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Return mean RMSE (log space) across TimeSeriesSplit folds."""
        tscv   = TimeSeriesSplit(n_splits=self.config.n_cv_splits)
        scores = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
            scores.append(rmse)
        return float(np.mean(scores))

    # ── Prophet baseline ──────────────────────────────────────────────────────

    def _run_prophet_baseline(self) -> dict:
        """
        Fit Prophet on daily aggregated Total Cost (INR) from train CSV.
        Returns test-set metrics for comparison only — not used as champion.
        """
        try:
            from prophet import Prophet  # lazy import — heavy dependency

            train_df = pd.read_csv(self.config.train_csv_path)
            train_df["ds"] = pd.to_datetime(
                train_df["Usage Start Date"], errors="coerce"
            ).dt.normalize()
            train_df["y"] = np.log1p(train_df["Total Cost (INR)"])

            daily = (
                train_df.dropna(subset=["ds"])
                .groupby("ds")["y"]
                .mean()
                .reset_index()
            )

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
            )
            model.fit(daily)

            # Forecast 30 days ahead as a sanity check
            future   = model.make_future_dataframe(periods=30, freq="D")
            forecast = model.predict(future)

            prophet_rmse = float(
                np.sqrt(mean_squared_error(daily["y"], forecast["yhat"][: len(daily)]))
            )

            logger.info(f"Prophet baseline train RMSE (log space): {prophet_rmse:.4f}")

            with mlflow.start_run(run_name="prophet_baseline", nested=False):
                mlflow.log_param("model_type", "Prophet")
                mlflow.log_param("yearly_seasonality", True)
                mlflow.log_param("weekly_seasonality", True)
                mlflow.log_metric("train_rmse_log", prophet_rmse)

            return {"prophet_train_rmse_log": round(prophet_rmse, 6)}

        except ImportError:
            logger.warning("Prophet not installed — skipping baseline. "
                           "Run: pip install prophet")
            return {"prophet_train_rmse_log": None}
        except Exception as e:
            logger.warning(f"Prophet baseline failed (non-critical): {e}")
            return {"prophet_train_rmse_log": None}

    # ── Optuna objective factories ─────────────────────────────────────────────

    def _xgb_objective(
        self, X: np.ndarray, y: np.ndarray
    ):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state":      self.config.random_state,
                "tree_method":       "hist",
                "verbosity":         0,
            }
            model = xgb.XGBRegressor(**params)
            return self._cv_score(model, X, y)
        return objective

    def _lgb_objective(
        self, X: np.ndarray, y: np.ndarray
    ):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 100),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state":     self.config.random_state,
                "verbose":          -1,
            }
            model = lgb.LGBMRegressor(**params)
            return self._cv_score(model, X, y)
        return objective

    # ── Train & log a single model ────────────────────────────────────────────

    def _train_and_log(
        self,
        model_name: str,
        model,
        params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_rmse: float,
    ) -> dict:
        """
        Fit the final model on full train set, evaluate on test,
        log everything to MLflow, return test metrics.
        """
        try:
            with mlflow.start_run(run_name=model_name) as run:
                # Fit on full train
                model.fit(X_train, y_train)

                # Predict
                train_preds = model.predict(X_train)
                test_preds  = model.predict(X_test)

                # Metrics
                train_metrics = compute_metrics(y_train, train_preds, "train_")
                test_metrics  = compute_metrics(y_test,  test_preds,  "test_")
                all_metrics   = {**train_metrics, **test_metrics, "cv_rmse": round(cv_rmse, 6)}

                # Log to MLflow
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(params)
                mlflow.log_metrics(all_metrics)

                # Feature importance plot
                if hasattr(model, "feature_importances_"):
                    fi = pd.DataFrame({
                        "feature":    self.feature_names,
                        "importance": model.feature_importances_,
                    }).sort_values("importance", ascending=False)

                    fi_path = os.path.join("artifacts", f"feature_importance_{model_name}.csv")
                    fi.to_csv(fi_path, index=False)
                    mlflow.log_artifact(fi_path)

                    logger.info(
                        f"[{model_name}] Top 5 features: "
                        + ", ".join(fi["feature"].head(5).tolist())
                    )

                # Log model
                mlflow.sklearn.log_model(model, artifact_path="model")

                run_id = run.info.run_id
                logger.info(
                    f"[{model_name}] MLflow run {run_id[:8]} — "
                    f"test MAPE: {test_metrics['test_mape']:.2f}%  "
                    f"test R²: {test_metrics['test_r2']:.4f}  "
                    f"CV RMSE: {cv_rmse:.4f}"
                )

            return {
                "model_name":  model_name,
                "run_id":      run_id,
                "params":      params,
                "metrics":     all_metrics,
            }

        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Optuna tuning ─────────────────────────────────────────────────────────

    def _tune_model(
        self,
        model_name: str,
        objective_fn,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[dict, float]:
        """Run Optuna study and return (best_params, best_cv_rmse)."""
        try:
            logger.info(
                f"Tuning {model_name} — {self.config.optuna_trials} trials ..."
            )
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
            )
            study.optimize(
                objective_fn(X, y),
                n_trials=self.config.optuna_trials,
                show_progress_bar=False,
            )
            best_params = study.best_params
            best_rmse   = study.best_value
            logger.info(
                f"{model_name} best CV RMSE: {best_rmse:.4f} | "
                f"best params: {best_params}"
            )
            return best_params, best_rmse
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Save champion ─────────────────────────────────────────────────────────

    def _save_champion(self, model, result: dict) -> None:
        try:
            joblib.dump(model, self.config.model_path)
            logger.info(f"Champion model saved -> {self.config.model_path}")

            report = {
                "champion_model":  result["model_name"],
                "run_id":          result["run_id"],
                "params":          result["params"],
                "metrics":         result["metrics"],
                "model_path":      self.config.model_path,
                "feature_count":   len(self.feature_names),
                "feature_names":   self.feature_names,
            }
            with open(self.config.model_report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Model report saved -> {self.config.model_report_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _register_model(self, run_id: str, model_name: str) -> None:
        """Register champion model in local MLflow Model Registry."""
        try:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=self.config.registered_model_name,
            )
            logger.info(
                f"Model registered as '{self.config.registered_model_name}' "
                f"version {mv.version} (run: {run_id[:8]})"
            )
        except Exception as e:
            logger.warning(f"Model registration failed (non-critical): {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def initiate_model_training(self) -> tuple[dict, str]:
        """
        Full training pipeline:
          1. Prophet baseline (benchmark)
          2. XGBoost — Optuna tuned
          3. LightGBM — Optuna tuned
          4. Select champion by lowest test MAPE
          5. Save champion model + report

        Returns
        -------
        tuple[dict, str]
            (test_metrics_dict, model_path)
        """
        logger.info("══════════════════════════════════════════════════")
        logger.info("  MODEL TRAINING — started")
        logger.info("══════════════════════════════════════════════════")

        try:
            # ── Setup ─────────────────────────────────────────────────────
            self._setup_mlflow()
            X_train, y_train, X_test, y_test = self._load_arrays()

            results = []

            # ── 1. Prophet baseline ───────────────────────────────────────
            logger.info("Running Prophet baseline ...")
            self._run_prophet_baseline()

            # ── 2. XGBoost ────────────────────────────────────────────────
            logger.info("Tuning XGBoost ...")
            xgb_params, xgb_cv_rmse = self._tune_model(
                "XGBoost", self._xgb_objective, X_train, y_train
            )
            xgb_model = xgb.XGBRegressor(
                **xgb_params,
                random_state=self.config.random_state,
                tree_method="hist",
                verbosity=0,
            )
            xgb_result = self._train_and_log(
                "XGBoost", xgb_model, xgb_params,
                X_train, y_train, X_test, y_test, xgb_cv_rmse,
            )
            results.append((xgb_model, xgb_result))

            # ── 3. LightGBM ───────────────────────────────────────────────
            logger.info("Tuning LightGBM ...")
            lgb_params, lgb_cv_rmse = self._tune_model(
                "LightGBM", self._lgb_objective, X_train, y_train
            )
            lgb_model = lgb.LGBMRegressor(
                **lgb_params,
                random_state=self.config.random_state,
                verbose=-1,
            )
            lgb_result = self._train_and_log(
                "LightGBM", lgb_model, lgb_params,
                X_train, y_train, X_test, y_test, lgb_cv_rmse,
            )
            results.append((lgb_model, lgb_result))

            # ── 4. Select champion by lowest test MAPE ────────────────────
            champion_model, champion_result = min(
                results,
                key=lambda r: r[1]["metrics"].get("test_mape", float("inf")),
            )

            logger.info(
                f"Champion: {champion_result['model_name']} — "
                f"test MAPE: {champion_result['metrics']['test_mape']:.2f}%  "
                f"test R²: {champion_result['metrics']['test_r2']:.4f}"
            )

            # ── 5. Save & register ────────────────────────────────────────
            # Re-fit champion on full train to get final model
            champion_model.fit(X_train, y_train)
            self._save_champion(champion_model, champion_result)
            self._register_model(
                champion_result["run_id"],
                champion_result["model_name"],
            )

            # ── Summary ───────────────────────────────────────────────────
            logger.info("── Model comparison ─────────────────────────────")
            for _, res in results:
                m = res["metrics"]
                logger.info(
                    f"  {res['model_name']:<12} | "
                    f"MAPE: {m['test_mape']:.2f}%  "
                    f"R²: {m['test_r2']:.4f}  "
                    f"RMSE(INR): {m['test_rmse_inr']:,.0f}"
                )
            logger.info("─────────────────────────────────────────────────")
            logger.info("  MODEL TRAINING — completed successfully")
            logger.info("══════════════════════════════════════════════════")

            return champion_result["metrics"], self.config.model_path

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise CustomException(e, sys) from e


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics, model_path = trainer.initiate_model_training()

    print("\nTraining complete.")
    print(f"  Champion model  : {json.load(open('artifacts/model_report.json'))['champion_model']}")
    print(f"  Test MAPE       : {metrics['test_mape']:.2f}%")
    print(f"  Test R²         : {metrics['test_r2']:.4f}")
    print(f"  Test RMSE (INR) : {metrics['test_rmse_inr']:,.0f}")
    print(f"  Model saved     : {model_path}")
    print("  MLflow UI       : mlflow ui --backend-store-uri ./mlruns")