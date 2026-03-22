# =============================================================================
# Makefile
# Developer shortcuts for the GCP Billing Forecaster project.
#
# Usage:
#   make help          list all targets
#   make install       install Python dependencies
#   make run           run Flask app locally (no Docker)
#   make pipeline      run full training pipeline locally
#   make build         build Docker image
#   make up            start Docker Compose stack
#   make down          stop Docker Compose stack
#   make test          run pytest suite
#   make lint          run ruff linter
#   make mlflow        open MLflow UI locally
#   make clean         remove artifacts and logs
# =============================================================================

PYTHON     = python
PYTHONPATH = PYTHONPATH=.
PORT       = 5000
IMAGE_NAME = gcp-billing-forecaster

.PHONY: help install run pipeline build up down logs test lint mlflow clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "GCP Billing Forecaster — available commands:"
	@echo ""
	@echo "  make install    Install all Python dependencies"
	@echo "  make run        Run Flask app locally (port $(PORT))"
	@echo "  make pipeline   Run full training pipeline end-to-end"
	@echo "  make build      Build Docker image"
	@echo "  make up         Start full stack with Docker Compose"
	@echo "  make down       Stop Docker Compose stack"
	@echo "  make logs       Tail app container logs"
	@echo "  make test       Run pytest test suite"
	@echo "  make lint       Run ruff linter"
	@echo "  make mlflow     Start MLflow UI locally (port 5001)"
	@echo "  make clean      Remove generated artifacts and logs"
	@echo ""

# ── Local development ─────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

run:
	$(PYTHONPATH) $(PYTHON) app.py

pipeline:
	$(PYTHONPATH) $(PYTHON) src/pipeline/training_pipeline.py

predict-test:
	$(PYTHONPATH) $(PYTHON) src/pipeline/prediction_pipeline.py

# ── Testing & linting ─────────────────────────────────────────────────────────
test:
	$(PYTHONPATH) pytest tests/ -v --tb=short

lint:
	ruff check src/ app.py

lint-fix:
	ruff check src/ app.py --fix

# ── Docker ────────────────────────────────────────────────────────────────────
build:
	docker build -t $(IMAGE_NAME):latest .

build-no-cache:
	docker build --no-cache -t $(IMAGE_NAME):latest .

up:
	docker compose up --build

up-detached:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f app

shell:
	docker compose exec app /bin/bash

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5001

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf artifacts/*.csv artifacts/*.npy artifacts/*.pkl artifacts/*.json
	rm -rf logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned artifacts, logs, and pycache."

clean-all: clean
	rm -rf mlruns/
	@echo "Also removed mlruns/."