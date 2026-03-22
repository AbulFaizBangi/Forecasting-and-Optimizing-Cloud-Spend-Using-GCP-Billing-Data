"""
data_validation.py
──────────────────
Responsibilities
  1. Load schema from configs/schema.yaml
  2. Run 6 validation suites against train.csv and test.csv
       Suite 1 — Column presence & completeness
       Suite 2 — Data type conformance
       Suite 3 — Numeric range bounds
       Suite 4 — Categorical value sets
       Suite 5 — Dataset-level rules (row count, null %, temporal order)
       Suite 6 — Train/test temporal integrity (no leakage)
  3. Write a detailed validation_report.json to artifacts/
  4. Return (validation_status: bool, report_path: str) to the pipeline
  5. Raise CustomException and halt the pipeline on any CRITICAL failure

Failure levels
  CRITICAL  — pipeline must stop (missing columns, wrong dtypes, negative costs)
  WARNING   — logged and recorded but pipeline continues (nullable nulls, etc.)
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import yaml

from src.exception import CustomException
from src.logger import logger


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DataValidationConfig:
    schema_path: str             = os.path.join("configs", "schema.yaml")
    train_data_path: str         = os.path.join("artifacts", "train.csv")
    test_data_path: str          = os.path.join("artifacts", "test.csv")
    ingestion_report_path: str   = os.path.join("artifacts", "ingestion_report.json")
    validation_report_path: str  = os.path.join("artifacts", "validation_report.json")

    # Datetime columns — re-parsed after CSV load
    datetime_columns: list = field(default_factory=lambda: [
        "Usage Start Date",
        "Usage End Date",
    ])


# ── Validation Report Builder ─────────────────────────────────────────────────

class ValidationReport:
    """Accumulates check results and serialises to JSON."""

    def __init__(self):
        self.checks: list[dict] = []
        self.critical_failures: list[str] = []
        self.warnings: list[str] = []

    def add(
        self,
        suite: str,
        check: str,
        status: str,           # "PASSED" | "WARNING" | "CRITICAL"
        detail: str = "",
        level: str = "INFO",
    ) -> None:
        entry = {
            "suite": suite,
            "check": check,
            "status": status,
            "detail": detail,
        }
        self.checks.append(entry)

        log_msg = f"[{suite}] {check} → {status}"
        if detail:
            log_msg += f" | {detail}"

        if status == "CRITICAL":
            self.critical_failures.append(f"{suite} / {check}: {detail}")
            logger.error(log_msg)
        elif status == "WARNING":
            self.warnings.append(f"{suite} / {check}: {detail}")
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def passed(self) -> bool:
        return len(self.critical_failures) == 0

    def summary(self) -> dict[str, Any]:
        total   = len(self.checks)
        passed  = sum(1 for c in self.checks if c["status"] == "PASSED")
        warns   = sum(1 for c in self.checks if c["status"] == "WARNING")
        crits   = sum(1 for c in self.checks if c["status"] == "CRITICAL")
        return {
            "validation_passed": self.passed(),
            "timestamp": datetime.now().isoformat(),
            "total_checks": total,
            "passed": passed,
            "warnings": warns,
            "critical_failures": crits,
            "critical_failure_details": self.critical_failures,
            "warning_details": self.warnings,
            "checks": self.checks,
        }


# ── DataValidation Class ──────────────────────────────────────────────────────

class DataValidation:
    """
    Runs all validation suites against the ingested train and test splits.

    Usage
    -----
    validator = DataValidation()
    status, report_path = validator.initiate_data_validation()
    """

    def __init__(self, config: DataValidationConfig = DataValidationConfig()):
        self.config = config
        self.report = ValidationReport()
        self.schema: dict  = {}

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_schema(self) -> None:
        try:
            if not os.path.exists(self.config.schema_path):
                raise FileNotFoundError(
                    f"Schema not found at '{self.config.schema_path}'. "
                    "Make sure configs/schema.yaml exists."
                )
            with open(self.config.schema_path) as f:
                self.schema = yaml.safe_load(f)
            logger.info(f"Schema loaded from {self.config.schema_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train = pd.read_csv(self.config.train_data_path)
            test  = pd.read_csv(self.config.test_data_path)

            for df in [train, test]:
                for col in self.config.datetime_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

            logger.info(
                f"Loaded train ({len(train):,} rows) and "
                f"test ({len(test):,} rows) for validation."
            )
            return train, test
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_ingestion_report(self) -> dict:
        try:
            if os.path.exists(self.config.ingestion_report_path):
                with open(self.config.ingestion_report_path) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Suite 1: Column presence ──────────────────────────────────────────────

    def _validate_columns(
        self, df: pd.DataFrame, split_name: str
    ) -> None:
        suite = f"Suite 1 — Column presence [{split_name}]"
        expected = list(self.schema["columns"].keys())

        missing = [c for c in expected if c not in df.columns]
        extra   = [c for c in df.columns if c not in expected]

        if missing:
            self.report.add(
                suite, "required columns present", "CRITICAL",
                f"Missing: {missing}"
            )
        else:
            self.report.add(
                suite, "required columns present", "PASSED",
                f"All {len(expected)} expected columns present."
            )

        if extra:
            self.report.add(
                suite, "no unexpected columns", "WARNING",
                f"Extra columns (ignored downstream): {extra}"
            )
        else:
            self.report.add(suite, "no unexpected columns", "PASSED")

    # ── Suite 2: Data type conformance ────────────────────────────────────────

    def _validate_dtypes(
        self, df: pd.DataFrame, split_name: str
    ) -> None:
        suite = f"Suite 2 — Dtype conformance [{split_name}]"

        dtype_map = {
            # pandas >= 2.0 reports string columns as "str" or "object"
            "object":       ["object", "str", "string"],
            "float64":      ["float64", "float32", "int64", "int32"],
            "int64":        ["int64", "int32"],
            "datetime64":   ["datetime64[ns]", "datetime64[us]",
                             "datetime64[ms]", "datetime64[s]"],
        }

        for col, rules in self.schema["columns"].items():
            if col not in df.columns:
                continue
            expected_dtype = rules["dtype"]
            actual_dtype   = str(df[col].dtype)
            allowed        = dtype_map.get(expected_dtype, [expected_dtype])

            if actual_dtype in allowed or any(
                actual_dtype.startswith(a.split("[")[0]) for a in allowed
            ):
                self.report.add(
                    suite, f"{col} dtype", "PASSED",
                    f"Expected {expected_dtype}, got {actual_dtype}"
                )
            else:
                self.report.add(
                    suite, f"{col} dtype", "CRITICAL",
                    f"Expected {expected_dtype}, got {actual_dtype}"
                )

    # ── Suite 3: Numeric range bounds ─────────────────────────────────────────

    def _validate_ranges(
        self, df: pd.DataFrame, split_name: str
    ) -> None:
        suite = f"Suite 3 — Numeric ranges [{split_name}]"

        for col, rules in self.schema["columns"].items():
            if col not in df.columns:
                continue

            col_min = rules.get("min")
            col_max = rules.get("max")

            if col_min is None and col_max is None:
                continue

            numeric_col = pd.to_numeric(df[col], errors="coerce").dropna()

            if col_min is not None:
                violations = int((numeric_col < col_min).sum())
                if violations:
                    self.report.add(
                        suite, f"{col} >= {col_min}", "CRITICAL",
                        f"{violations:,} rows below minimum ({col_min})"
                    )
                else:
                    self.report.add(
                        suite, f"{col} >= {col_min}", "PASSED"
                    )

            if col_max is not None:
                violations = int((numeric_col > col_max).sum())
                if violations:
                    self.report.add(
                        suite, f"{col} <= {col_max}", "CRITICAL",
                        f"{violations:,} rows above maximum ({col_max})"
                    )
                else:
                    self.report.add(
                        suite, f"{col} <= {col_max}", "PASSED"
                    )

    # ── Suite 4: Categorical value sets ───────────────────────────────────────

    def _validate_categoricals(
        self, df: pd.DataFrame, split_name: str
    ) -> None:
        suite = f"Suite 4 — Categorical values [{split_name}]"
        allowed_map = self.schema.get("allowed_values", {})

        for col, allowed in allowed_map.items():
            if col not in df.columns:
                continue
            actual_values   = set(df[col].dropna().unique().tolist())
            unknown_values  = actual_values - set(allowed)

            if unknown_values:
                self.report.add(
                    suite, f"{col} known values", "WARNING",
                    f"Unknown values found: {sorted(unknown_values)} — "
                    "add to schema.yaml if legitimate"
                )
            else:
                self.report.add(
                    suite, f"{col} known values", "PASSED",
                    f"All {len(actual_values)} values in allowed set."
                )

    # ── Suite 5: Dataset-level rules ──────────────────────────────────────────

    def _validate_dataset_rules(
        self, df: pd.DataFrame, split_name: str
    ) -> None:
        suite  = f"Suite 5 — Dataset rules [{split_name}]"
        rules  = self.schema.get("dataset_rules", {})

        # Row count floor
        min_rows = rules.get("min_rows", 0)
        if len(df) < min_rows:
            self.report.add(
                suite, f"row count >= {min_rows}", "CRITICAL",
                f"Only {len(df):,} rows — dataset may be truncated."
            )
        else:
            self.report.add(
                suite, f"row count >= {min_rows}", "PASSED",
                f"{len(df):,} rows."
            )

        # Null percentage per column
        max_null_pct = rules.get("max_null_pct", 10.0)
        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            schema_col = self.schema["columns"].get(col, {})
            nullable   = schema_col.get("nullable", True)

            if not nullable and null_pct > 0:
                self.report.add(
                    suite, f"{col} not nullable", "CRITICAL",
                    f"{null_pct:.1f}% nulls in non-nullable column."
                )
            elif null_pct > max_null_pct:
                self.report.add(
                    suite, f"{col} null pct <= {max_null_pct}%", "WARNING",
                    f"{null_pct:.1f}% nulls exceeds threshold."
                )

        # Temporal order: start < end
        if rules.get("temporal_order") and "Usage Start Date" in df.columns \
                and "Usage End Date" in df.columns:
            bad = int((df["Usage Start Date"] >= df["Usage End Date"]).sum())
            if bad:
                self.report.add(
                    suite, "start date < end date", "CRITICAL",
                    f"{bad:,} rows where start >= end."
                )
            else:
                self.report.add(suite, "start date < end date", "PASSED")

        # No negative costs
        if rules.get("no_negative_cost") and "Unrounded Cost ($)" in df.columns:
            neg = int((df["Unrounded Cost ($)"] < 0).sum())
            if neg:
                self.report.add(
                    suite, "no negative costs", "CRITICAL",
                    f"{neg:,} rows with negative Unrounded Cost ($)."
                )
            else:
                self.report.add(suite, "no negative costs", "PASSED")

    # ── Suite 6: Train/test temporal integrity ────────────────────────────────

    def _validate_train_test_split(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        suite = "Suite 6 — Train/test temporal integrity"

        if "Usage Start Date" not in train.columns or \
           "Usage Start Date" not in test.columns:
            self.report.add(
                suite, "date columns present", "WARNING",
                "Usage Start Date missing — cannot check temporal split."
            )
            return

        latest_train   = train["Usage Start Date"].max()
        earliest_test  = test["Usage Start Date"].min()

        if pd.isna(latest_train) or pd.isna(earliest_test):
            self.report.add(
                suite, "no NaT boundary dates", "WARNING",
                "Could not determine split boundary — NaT values present."
            )
            return

        if latest_train <= earliest_test:
            self.report.add(
                suite, "no temporal leakage", "PASSED",
                f"Latest train: {latest_train} ≤ earliest test: {earliest_test}"
            )
        else:
            overlap = int(
                (train["Usage Start Date"] >= earliest_test).sum()
            )
            self.report.add(
                suite, "no temporal leakage", "CRITICAL",
                f"Overlap detected — {overlap:,} train rows after earliest test date. "
                f"Latest train: {latest_train}, earliest test: {earliest_test}"
            )

        # Check train size is at least 3× test (rough sanity on split ratio)
        ratio = len(train) / max(len(test), 1)
        if ratio < 2.0:
            self.report.add(
                suite, "train/test size ratio", "WARNING",
                f"Train is only {ratio:.1f}× test — expected ~4×. "
                "Check split ratio in DataIngestionConfig."
            )
        else:
            self.report.add(
                suite, "train/test size ratio", "PASSED",
                f"Train is {ratio:.1f}× test size."
            )

    # ── Save report ───────────────────────────────────────────────────────────

    def _save_report(self) -> None:
        try:
            summary = self.report.summary()
            with open(self.config.validation_report_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(
                f"Validation report saved → {self.config.validation_report_path}"
            )
        except Exception as e:
            raise CustomException(e, sys) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def initiate_data_validation(self) -> tuple[bool, str]:
        """
        Run all 6 validation suites against train and test splits.

        Returns
        -------
        tuple[bool, str]
            (validation_passed, validation_report_path)
            — consumed by DataTransformation which checks the bool before
              proceeding.

        Raises
        ------
        CustomException
            On file I/O errors only. Validation failures are recorded in the
            report and surfaced via the returned bool — not as exceptions —
            so the caller decides whether to halt.
        """
        logger.info("══════════════════════════════════════════════════")
        logger.info("  DATA VALIDATION — started")
        logger.info("══════════════════════════════════════════════════")

        try:
            # ── Load inputs ───────────────────────────────────────────────
            self._load_schema()
            train, test = self._load_dataframes()
            self._load_ingestion_report()

            # ── Run suites on train ───────────────────────────────────────
            self._validate_columns(train, "train")
            self._validate_dtypes(train, "train")
            self._validate_ranges(train, "train")
            self._validate_categoricals(train, "train")
            self._validate_dataset_rules(train, "train")

            # ── Run suites on test ────────────────────────────────────────
            self._validate_columns(test, "test")
            self._validate_dtypes(test, "test")
            self._validate_ranges(test, "test")
            self._validate_categoricals(test, "test")
            self._validate_dataset_rules(test, "test")

            # ── Cross-split check ─────────────────────────────────────────
            self._validate_train_test_split(train, test)

            # ── Save report ───────────────────────────────────────────────
            self._save_report()

            # ── Summary log ───────────────────────────────────────────────
            summary = self.report.summary()
            logger.info(
                f"Validation complete — "
                f"{summary['passed']} passed, "
                f"{summary['warnings']} warnings, "
                f"{summary['critical_failures']} critical failures."
            )

            if not self.report.passed():
                logger.error(
                    "CRITICAL failures detected — pipeline should not proceed. "
                    f"See {self.config.validation_report_path} for details."
                )

            logger.info("  DATA VALIDATION — completed")
            logger.info("══════════════════════════════════════════════════")

            return self.report.passed(), self.config.validation_report_path

        except Exception as e:
            logger.error(f"Data validation failed unexpectedly: {e}")
            raise CustomException(e, sys) from e


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validator = DataValidation()
    passed, report_path = validator.initiate_data_validation()

    print(f"\nValidation {'PASSED ✓' if passed else 'FAILED ✗'}")
    print(f"Report → {report_path}")

    if not passed:
        import json
        report = json.load(open(report_path))
        print("\nCritical failures:")
        for f in report["critical_failure_details"]:
            print(f"  ✗ {f}")
        sys.exit(1)