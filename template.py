import os
from pathlib import Path

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",   # Fetching data
    "src/components/data_validation.py",  # Checking data quality
    "src/components/data_transformation.py", # Cleaning & Processing
    "src/components/model_trainer.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "app.py",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass