# Screenshots Directory

This directory contains screenshots for the project README.

## Required Screenshots

Please add the following screenshots to this directory:

### 1. Training Pipeline (`training_pipeline.png`)
- **Command**: `python src/pipeline/training_pipeline.py`
- **What to capture**: Terminal output showing:
  - Data ingestion progress
  - Feature engineering steps
  - Optuna hyperparameter tuning trials
  - Model training metrics (RMSE, MAE, MAPE, R²)
  - Champion model selection

### 2. MLflow UI (`mlflow_ui.png`)
- **Command**: `mlflow ui --backend-store-uri ./mlruns`
- **What to capture**: Browser window showing:
  - Experiment list with XGBoost and LightGBM runs
  - Metrics comparison table
  - Run details with parameters and metrics
  - Feature importance artifacts

### 3. Flask Application (`flask_app.png`)
- **Command**: `python app.py`
- **What to capture**: Browser window at http://localhost:5000 showing:
  - Home page with input form
  - Service name dropdown
  - Input fields for usage metrics
  - Predict button

### 4. Compute Engine Input (`compute_engine_input.png`)
- **What to capture**: Input form filled with Compute Engine parameters:
  - Service Name: "Compute Engine"
  - Region/Zone: "us-central1"
  - Usage Quantity: 100.0
  - CPU Utilization: 75.0%
  - Memory Utilization: 60.0%

### 5. Compute Engine Result (`compute_engine_result.png`)
- **What to capture**: Prediction result page showing:
  - Predicted cost in INR
  - Model used (XGBoost/LightGBM)
  - Timestamp
  - Input parameters summary

### 6. Cloud Storage Input (`cloud_storage_input.png`)
- **What to capture**: Input form filled with Cloud Storage parameters:
  - Service Name: "Cloud Storage"
  - Region/Zone: "us-central1"
  - Usage Quantity: 500.0
  - Network metrics

### 7. Cloud Storage Result (`cloud_storage_result.png`)
- **What to capture**: Prediction result for Cloud Storage showing:
  - Predicted storage cost
  - Model metadata
  - Timestamp

## Screenshot Guidelines

- **Format**: PNG (preferred) or JPG
- **Resolution**: At least 1280x720 for clarity
- **File naming**: Use exact names listed above (lowercase, underscores)
- **Content**: Ensure sensitive information is not visible
- **Quality**: Clear, readable text and UI elements

## How to Take Screenshots

### macOS
- Full screen: `Cmd + Shift + 3`
- Selected area: `Cmd + Shift + 4`
- Window: `Cmd + Shift + 4`, then press `Space`

### Windows
- Full screen: `PrtScn` or `Win + PrtScn`
- Selected area: `Win + Shift + S`
- Snipping Tool: Search for "Snipping Tool" in Start menu

### Linux
- Full screen: `PrtScn`
- Selected area: `Shift + PrtScn`
- GNOME Screenshot: `gnome-screenshot`

## After Adding Screenshots

1. Verify all 7 screenshots are in this directory
2. Check that filenames match exactly
3. Commit and push to GitHub:
   ```bash
   git add screenshots/
   git commit -m "Add project screenshots"
   git push origin main
   ```

The screenshots will automatically appear in the README.md file!
