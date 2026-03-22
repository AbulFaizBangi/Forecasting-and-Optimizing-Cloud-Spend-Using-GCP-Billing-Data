"""
app.py
──────
Flask web application for GCP Cloud Billing Cost Forecasting.

Routes
------
GET  /          → index.html  — prediction form
POST /predict   → results.html — shows forecasted cost
GET  /health    → JSON liveness probe (used by Docker / Kubernetes)
GET  /metrics   → Prometheus-compatible plaintext metrics
GET  /api/predict → JSON API endpoint (for programmatic access)
"""

import json
import os
import sys
import time
from collections import defaultdict

from flask import Flask, jsonify, render_template, request

from src.exception import CustomException
from src.logger import logger
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "gcp-billing-mlops-dev")

# ── Load pipeline once at startup ─────────────────────────────────────────────

pipeline = PredictPipeline()

# ── Metrics counters ──────────────────────────────────────────────────────────

_metrics = defaultdict(float)
_metrics["requests_total"]    = 0
_metrics["predictions_total"] = 0
_metrics["errors_total"]      = 0
_start_time                   = time.time()

# ── Dropdown data ─────────────────────────────────────────────────────────────

SERVICES = [
    "App Engine", "Artifact Registry", "BigQuery", "Cloud Armor",
    "Cloud Build", "Cloud CDN", "Cloud Dataflow", "Cloud Dataproc",
    "Cloud Filestore", "Cloud Functions", "Cloud Logging",
    "Cloud Memorystore", "Cloud Monitoring", "Cloud Pub/Sub",
    "Cloud Run", "Cloud SQL", "Cloud Scheduler", "Cloud Spanner",
    "Cloud Storage", "Cloud Tasks", "Compute Engine",
    "Dialogflow", "Secret Manager",
]

REGIONS = [
    "asia-east1", "asia-south1", "europe-west1",
    "us-central1", "us-east1", "us-west1",
]

UNITS = ["GB", "Hours", "Requests"]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Render the prediction form."""
    _metrics["requests_total"] += 1
    return render_template(
        "index.html",
        services=SERVICES,
        regions=REGIONS,
        units=UNITS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and return prediction results."""
    _metrics["requests_total"]    += 1
    _metrics["predictions_total"] += 1
    t_start = time.time()

    try:
        # ── Parse form inputs ─────────────────────────────────────────────
        features = CustomData(
            service_name           = request.form.get("service_name", "Cloud Run"),
            region_zone            = request.form.get("region_zone", "us-central1"),
            usage_quantity         = float(request.form.get("usage_quantity", 0)),
            usage_unit             = request.form.get("usage_unit", "Requests"),
            cpu_utilization        = float(request.form.get("cpu_utilization", 0)),
            memory_utilization     = float(request.form.get("memory_utilization", 0)),
            network_inbound_bytes  = float(request.form.get("network_inbound_bytes", 0)),
            network_outbound_bytes = float(request.form.get("network_outbound_bytes", 0)),
            cost_per_quantity      = float(request.form.get("cost_per_quantity", 0)),
            usage_start_date       = request.form.get("usage_start_date", ""),
            usage_end_date         = request.form.get("usage_end_date", ""),
        )

        # ── Run prediction ────────────────────────────────────────────────
        predicted_inr = pipeline.predict(features)
        predicted_usd = round(predicted_inr / 83.0, 2)
        latency_ms    = round((time.time() - t_start) * 1000, 1)

        _metrics["latency_ms_last"] = latency_ms

        logger.info(
            f"Prediction served — ₹{predicted_inr:,.0f} INR | "
            f"{latency_ms}ms | "
            f"service={features.service_name} region={features.region_zone}"
        )

        return render_template(
            "results.html",
            prediction_inr  = f"{predicted_inr:,.0f}",
            prediction_usd  = f"{predicted_usd:,.2f}",
            service_name    = features.service_name,
            region_zone     = features.region_zone,
            usage_quantity  = features.usage_quantity,
            usage_unit      = features.usage_unit,
            latency_ms      = latency_ms,
            services        = SERVICES,
            regions         = REGIONS,
            units           = UNITS,
            form_data       = request.form,
        )

    except Exception as e:
        _metrics["errors_total"] += 1
        logger.error(f"Prediction error: {e}")
        return render_template(
            "index.html",
            error    = f"Prediction failed: {str(e)}",
            services = SERVICES,
            regions  = REGIONS,
            units    = UNITS,
        ), 400


@app.route("/health", methods=["GET"])
def health():
    """
    Liveness probe — used by Docker HEALTHCHECK and Kubernetes.
    Returns 200 if model is loaded, 503 if not.
    """
    try:
        info   = pipeline.model_info
        uptime = round(time.time() - _start_time, 1)
        return jsonify({
            "status":      "ok",
            "model":       info.get("model_type", "unknown"),
            "n_features":  info.get("n_features", 0),
            "uptime_s":    uptime,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 503


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Prometheus-compatible plaintext metrics endpoint.
    Scraped by Prometheus / Cloud Monitoring in production.
    """
    uptime = round(time.time() - _start_time, 1)
    lines  = [
        "# HELP gcp_billing_requests_total Total HTTP requests",
        "# TYPE gcp_billing_requests_total counter",
        f"gcp_billing_requests_total {int(_metrics['requests_total'])}",
        "",
        "# HELP gcp_billing_predictions_total Total predictions served",
        "# TYPE gcp_billing_predictions_total counter",
        f"gcp_billing_predictions_total {int(_metrics['predictions_total'])}",
        "",
        "# HELP gcp_billing_errors_total Total prediction errors",
        "# TYPE gcp_billing_errors_total counter",
        f"gcp_billing_errors_total {int(_metrics['errors_total'])}",
        "",
        "# HELP gcp_billing_uptime_seconds App uptime in seconds",
        "# TYPE gcp_billing_uptime_seconds gauge",
        f"gcp_billing_uptime_seconds {uptime}",
        "",
        "# HELP gcp_billing_latency_ms_last Latency of last prediction in ms",
        "# TYPE gcp_billing_latency_ms_last gauge",
        f"gcp_billing_latency_ms_last {_metrics['latency_ms_last']}",
    ]
    return "\n".join(lines), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint — accepts and returns JSON.
    Used for programmatic access and testing.

    Request body (JSON)
    -------------------
    {
        "service_name": "Cloud Run",
        "region_zone": "us-central1",
        "usage_quantity": 500,
        "usage_unit": "Requests",
        "cpu_utilization": 72.5,
        "memory_utilization": 48.3,
        "network_inbound_bytes": 500000000,
        "network_outbound_bytes": 500000000,
        "cost_per_quantity": 4.5,
        "usage_start_date": "2022-06-15 14:00:00",
        "usage_end_date": "2022-06-15 22:00:00"
    }

    Response (JSON)
    ---------------
    {
        "predicted_cost_inr": 187036.0,
        "predicted_cost_usd": 2253.0,
        "service_name": "Cloud Run",
        "region_zone": "us-central1",
        "latency_ms": 12.3
    }
    """
    _metrics["requests_total"]    += 1
    _metrics["predictions_total"] += 1
    t_start = time.time()

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        features = CustomData(
            service_name           = data.get("service_name", "Cloud Run"),
            region_zone            = data.get("region_zone", "us-central1"),
            usage_quantity         = float(data.get("usage_quantity", 0)),
            usage_unit             = data.get("usage_unit", "Requests"),
            cpu_utilization        = float(data.get("cpu_utilization", 0)),
            memory_utilization     = float(data.get("memory_utilization", 0)),
            network_inbound_bytes  = float(data.get("network_inbound_bytes", 0)),
            network_outbound_bytes = float(data.get("network_outbound_bytes", 0)),
            cost_per_quantity      = float(data.get("cost_per_quantity", 0)),
            usage_start_date       = data.get("usage_start_date", ""),
            usage_end_date         = data.get("usage_end_date", ""),
        )

        predicted_inr = pipeline.predict(features)
        predicted_usd = round(predicted_inr / 83.0, 2)
        latency_ms    = round((time.time() - t_start) * 1000, 1)

        return jsonify({
            "predicted_cost_inr": predicted_inr,
            "predicted_cost_usd": predicted_usd,
            "service_name":       features.service_name,
            "region_zone":        features.region_zone,
            "latency_ms":         latency_ms,
        }), 200

    except Exception as e:
        _metrics["errors_total"] += 1
        logger.error(f"API prediction error: {e}")
        return jsonify({"error": str(e)}), 400


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting Flask app on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)