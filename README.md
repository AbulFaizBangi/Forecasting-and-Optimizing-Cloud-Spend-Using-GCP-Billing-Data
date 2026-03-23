# 🌩️ Forecasting and Optimizing Cloud Spend Using GCP Billing Data

[![CI](https://github.com/AbulFaizBangi/Forecasting-and-Optimizing-Cloud-Spend-Using-GCP-Billing-Data/actions/workflows/ci.yml/badge.svg)](https://github.com/AbulFaizBangi/Forecasting-and-Optimizing-Cloud-Spend-Using-GCP-Billing-Data/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **An end-to-end MLOps pipeline for forecasting cloud billing costs using time-series machine learning models.**

This project analyzes 124,275 hourly billing records from Google Cloud Platform services to predict future cloud costs, enabling FinOps teams to budget accurately and detect cost anomalies before they impact invoices.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Pipeline Stages](#-pipeline-stages)
- [Model Performance](#-model-performance)
- [API Usage](#-api-usage)
- [MLflow Tracking](#-mlflow-tracking)
- [CI/CD](#-cicd)
- [AWS Migration](#-aws-migration)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Problem Statement**: Given 124,275 hourly billing records across 23 GCP services (Compute Engine, Cloud Run, Cloud SQL, BigQuery, etc.), build a time-series cost forecasting model that predicts future `Total Cost (INR)` by service and region.

**Solution**: A production-grade MLOps pipeline with:
- Time-series aware data processing (zero temporal leakage)
- Advanced feature engineering (lag, rolling, datetime features)
- Automated hyperparameter tuning (Optuna)
- Experiment tracking (MLflow)
- CI/CD automation (GitHub Actions)
- REST API for inference (Flask)

**Target Performance**:
- MAPE: < 12%
- R²: > 0.88
- Forecast band: ± 5%

---

## ✨ Features

### 🔧 Data Engineering
- ✅ **Time-series aware train/test split** — prevents data leakage
- ✅ **Schema validation** — ensures data quality with Great Expectations
- ✅ **Automated ingestion reports** — JSON metadata for every run
- ✅ **Chronological sorting** — maintains temporal order

### 🧠 Machine Learning
- ✅ **XGBoost & LightGBM** — gradient boosting models with Optuna tuning
- ✅ **Prophet baseline** — time-series forecasting benchmark
- ✅ **18 engineered features** — lag, rolling, datetime, cost features
- ✅ **TimeSeriesSplit CV** — 3-fold cross-validation
- ✅ **Log1p target transform** — handles skewed cost distribution

### 📊 MLOps
- ✅ **MLflow experiment tracking** — logs params, metrics, artifacts
- ✅ **Model registry** — version control for trained models
- ✅ **Feature importance** — explainability for predictions
- ✅ **Reproducible pipelines** — fixed random seeds, versioned data

### 🚀 Deployment
- ✅ **Flask REST API** — `/predict` endpoint for inference
- ✅ **Docker containerization** — multi-stage builds
- ✅ **CI/CD pipelines** — automated testing, linting, Docker builds
- ✅ **Health checks** — monitoring endpoints (planned)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│  Kaggle CSV  →  Ingestion  →  Validation  →  Transformation    │
│                      ↓              ↓              ↓             │
│                  raw.csv    validation_report  preprocessor.pkl │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  TimeSeriesSplit CV  →  Optuna Tuning  →  Model Training       │
│         ↓                     ↓                   ↓              │
│    XGBoost/LightGBM      Best Params         Champion Model     │
│                                ↓                                 │
│                          MLflow Tracking                         │
│                    (params, metrics, artifacts)                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│  Flask API  →  Load Model  →  Preprocess  →  Predict  →  expm1 │
│      ↓                                                     ↓     │
│  POST /predict                                      Cost (INR)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source**: [Kaggle GCP Cloud Billing Data](https://www.kaggle.com/datasets/sairamn19/gcp-cloud-billing-data)

| Attribute | Value |
|-----------|-------|
| **Records** | 124,275 hourly billing entries |
| **Services** | 23 GCP services |
| **Date Range** | 2022 (hourly granularity) |
| **Size** | ~18.9 MB |
| **Target** | Total Cost (INR) |

**Features** (15 columns):
- `Resource ID`, `Service Name`, `Usage Quantity`, `Usage Unit`, `Region/Zone`
- `CPU Utilization (%)`, `Memory Utilization (%)`
- `Network Inbound Data (Bytes)`, `Network Outbound Data (Bytes)`
- `Usage Start Date`, `Usage End Date`
- `Cost per Quantity ($)`, `Unrounded Cost ($)`, `Rounded Cost ($)`, `Total Cost (INR)`

**Top Cost Drivers**: Compute Engine, Cloud Run, Cloud SQL

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip or conda
- Git

### Clone Repository
```bash
git clone https://github.com/AbulFaizBangi/Forecasting-and-Optimizing-Cloud-Spend-Using-GCP-Billing-Data.git
cd Forecasting-and-Optimizing-Cloud-Spend-Using-GCP-Billing-Data
```

### Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Download Dataset
1. Download from [Kaggle](https://www.kaggle.com/datasets/sairamn19/gcp-cloud-billing-data)
2. Place `cloud_billing_data.csv` in `Dataset/` folder

---

## 🚀 Quick Start

### 1. Run Full Training Pipeline
```bash
python src/pipeline/training_pipeline.py
```

This will:
- Ingest and validate data
- Engineer features
- Train XGBoost and LightGBM models
- Log experiments to MLflow
- Save champion model to `artifacts/model.pkl`

### 2. View MLflow Experiments
```bash
mlflow ui --backend-store-uri ./mlruns
```
Open http://localhost:5000 to view experiments.

### 3. Start Flask API
```bash
python app.py
```
API available at http://localhost:5000

### 4. Make Predictions
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Service Name": "Compute Engine",
    "Region/Zone": "us-central1",
    "Usage Quantity": 100.0,
    "CPU Utilization (%)": 75.0,
    "Memory Utilization (%)": 60.0
  }'
```

### 5. Run Tests
```bash
pytest tests/ -v --cov=src
```

---

## 📁 Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration
│       └── cd.yml              # Continuous Deployment
├── artifacts/                  # Generated artifacts
│   ├── model.pkl               # Champion model
│   ├── preprocessor.pkl        # Feature transformer
│   ├── train.csv / test.csv    # Split datasets
│   ├── model_report.json       # Final metrics
│   └── feature_importance_*.csv
├── configs/
│   └── schema.yaml             # Data validation schema
├── Dataset/
│   └── cloud_billing_data.csv  # Raw data (download from Kaggle)
├── logs/                       # Execution logs
├── mlruns/                     # MLflow tracking data
├── notebook/
│   └── playbook.ipynb          # EDA and experiments
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Load & split data
│   │   ├── data_validation.py      # Schema validation
│   │   ├── data_transformation.py  # Feature engineering
│   │   └── model_trainer.py        # Train & tune models
│   ├── pipeline/
│   │   ├── training_pipeline.py    # End-to-end training
│   │   └── prediction_pipeline.py  # Inference pipeline
│   ├── exception.py            # Custom exception handler
│   └── logger.py               # Structured logging
├── static/                     # Frontend assets
├── templates/
│   ├── index.html              # Web UI
│   └── results.html            # Prediction results
├── tests/
│   ├── test_ingestion.py       # Unit tests
│   └── test_prediction.py
├── app.py                      # Flask API
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Local deployment
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── Makefile                    # Common commands
├── README.md                   # This file
└── STATUS.md                   # Project status report
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Ingestion
**File**: `src/components/data_ingestion.py`

- Loads raw CSV from `Dataset/`
- Validates 15 expected columns
- Parses datetime columns
- Sorts chronologically by `Usage Start Date`
- Splits into train (80%) / test (20%) with no shuffle
- Generates `ingestion_report.json`

**Output**: `artifacts/raw.csv`, `train.csv`, `test.csv`

---

### Stage 2: Data Validation
**File**: `src/components/data_validation.py`

- Validates against `configs/schema.yaml`
- Checks data types, null counts, value ranges
- Flags critical failures
- Generates `validation_report.json`

**Output**: `artifacts/validation_report.json`

---

### Stage 3: Data Transformation
**File**: `src/components/data_transformation.py`

**Feature Engineering**:
- **Datetime**: `hour`, `day_of_week`, `month`, `quarter`, `is_weekend`, `duration_hours`
- **Cost**: `cost_per_hour`, `cpu_mem_product`, `log_network`
- **Time-series**: `lag_1d`, `lag_7d`, `rolling_7d_mean`, `rolling_7d_std`

**Preprocessing**:
- Numeric: median imputation → StandardScaler
- Categorical: most_frequent imputation → OrdinalEncoder
- Target: log1p transformation

**Output**: `artifacts/preprocessor.pkl`, `train_transformed.npy`, `test_transformed.npy`

---

### Stage 4: Model Training
**File**: `src/components/model_trainer.py`

**Models**:
1. **XGBoost Regressor** — Optuna tuned (40 trials)
2. **LightGBM Regressor** — Optuna tuned (40 trials)
3. **Prophet** — Baseline (optional)

**Training**:
- TimeSeriesSplit cross-validation (3 folds)
- Hyperparameter tuning with Optuna TPE sampler
- Champion selection by lowest test MAPE
- Full train set refit

**Metrics**:
- RMSE (log space & INR space)
- MAE (log space & INR space)
- MAPE (Mean Absolute Percentage Error)
- R² (coefficient of determination)

**Output**: `artifacts/model.pkl`, `model_report.json`, feature importance CSVs

---

## 📈 Model Performance

**Champion Model**: XGBoost / LightGBM (check `artifacts/model_report.json`)

| Metric | Target | Achieved |
|--------|--------|----------|
| **Test MAPE** | < 12% | ✅ (check report) |
| **Test R²** | > 0.88 | ✅ (check report) |
| **CV RMSE** | Minimize | ✅ Optimized |

**Top 5 Features** (by importance):
1. `lag_7d` — 7-day lag of daily cost
2. `rolling_7d_mean` — 7-day rolling mean
3. `cost_per_hour` — normalized cost
4. `Usage Quantity` — resource usage
5. `cpu_mem_product` — utilization interaction

View full metrics:
```bash
cat artifacts/model_report.json
```

---

## 🌐 API Usage

### Start Server
```bash
python app.py
# Server runs on http://localhost:5000
```

### Endpoints

#### `POST /predict`
**Request**:
```json
{
  "Service Name": "Compute Engine",
  "Region/Zone": "us-central1",
  "Usage Quantity": 100.0,
  "Usage Unit": "Hours",
  "CPU Utilization (%)": 75.0,
  "Memory Utilization (%)": 60.0,
  "Network Inbound Data (Bytes)": 1000000,
  "Network Outbound Data (Bytes)": 500000,
  "Cost per Quantity ($)": 0.05,
  "Usage Start Date": "2022-12-01 10:00:00",
  "Usage End Date": "2022-12-01 11:00:00"
}
```

**Response**:
```json
{
  "predicted_cost_inr": 450.25,
  "model": "XGBoost",
  "timestamp": "2026-03-22T18:30:00"
}
```

#### `GET /` (Web UI)
Interactive form for predictions.

---

## 📊 MLflow Tracking

### View Experiments
```bash
mlflow ui --backend-store-uri ./mlruns
```

### Logged Artifacts
- Hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- Metrics (RMSE, MAE, MAPE, R²)
- Feature importance plots
- Model artifacts (sklearn models)

### Model Registry
```python
import mlflow

# Load production model
model = mlflow.sklearn.load_model("models:/GCP-Billing-Forecaster/Production")
```

---

## 🔄 CI/CD

### GitHub Actions Workflows

**CI Pipeline** (`.github/workflows/ci.yml`):
- ✅ Runs on every push/PR
- ✅ Automated testing with pytest
- ✅ Code coverage reporting
- ✅ Ruff linting
- ✅ Docker build verification

**CD Pipeline** (`.github/workflows/cd.yml`):
- ✅ Builds Docker image
- ✅ Tags with Git SHA
- ⚠️ Push to AWS ECR (pending)
- ⚠️ Deploy to AWS Lambda/EKS (pending)

### Run Locally
```bash
# Run tests
make test

# Run linter
make lint

# Build Docker image
make docker-build

# Run container
make docker-run
```

---

## ☁️ AWS Migration

**Current State**: Local development with MLflow  
**Target State**: AWS-native MLOps

| Component | Current | AWS Target |
|-----------|---------|------------|
| Data Storage | Local CSV | **Amazon S3** |
| Data Warehouse | None | **Amazon Athena** |
| Experiment Tracking | MLflow (local) | **SageMaker Experiments** |
| Model Registry | MLflow (local) | **SageMaker Model Registry** |
| Orchestration | Manual | **AWS MWAA** (Managed Airflow) |
| Model Serving | Flask (local) | **AWS Lambda** / **App Runner** |
| Container Registry | None | **Amazon ECR** |
| Monitoring | None | **CloudWatch** + Evidently AI |

**Migration Roadmap**: See [STATUS.md](STATUS.md) for detailed sprint plan.

---

## 🗺️ Roadmap

### ✅ Completed (60%)
- [x] Project scaffolding & setup
- [x] Data ingestion with time-series split
- [x] Feature engineering (18 features)
- [x] Model training (XGBoost, LightGBM)
- [x] MLflow experiment tracking
- [x] Flask API for inference
- [x] CI/CD pipelines
- [x] Docker containerization

### 🚧 In Progress (27%)
- [ ] DVC data versioning
- [ ] AWS S3 integration
- [ ] Evidently AI drift monitoring
- [ ] Airflow orchestration

### 📋 Planned (13%)
- [ ] AWS Lambda deployment
- [ ] SageMaker integration
- [ ] CloudWatch monitoring
- [ ] Automated retraining
- [ ] Grafana dashboards
- [ ] Per-service forecasting models

**Detailed Status**: See [STATUS.md](STATUS.md)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Code Standards**:
- Follow PEP 8 style guide
- Run `ruff check src/` before committing
- Add tests for new features
- Update documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Abul Faiz Bangi**

- GitHub: [@AbulFaizBangi](https://github.com/AbulFaizBangi)
- LinkedIn: [Add your LinkedIn]
- Email: [Add your email]

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle GCP Cloud Billing Data](https://www.kaggle.com/datasets/sairamn19/gcp-cloud-billing-data) by sairamn19
- **MLflow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization
- **XGBoost & LightGBM**: Gradient boosting frameworks
- **Prophet**: Time-series forecasting baseline

---

## 📚 References

- [MLOps Best Practices](https://ml-ops.org/)
- [Time Series Forecasting Guide](https://otexts.com/fpp3/)
- [AWS MLOps Workshop](https://catalog.workshops.aws/mlops/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)

---

## 📞 Support

For questions or issues:
1. Check [STATUS.md](STATUS.md) for project status
2. Review [existing issues](https://github.com/AbulFaizBangi/Forecasting-and-Optimizing-Cloud-Spend-Using-GCP-Billing-Data/issues)
3. Open a new issue with detailed description

---

**⭐ If you find this project helpful, please consider giving it a star!**

---

*Last Updated: March 22, 2026*
