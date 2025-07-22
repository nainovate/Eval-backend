"""Prometheus metrics setup for FastAPI."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, Response
import time
import psutil
import os

# Define custom metrics
REQUEST_COUNT = Counter(
    'evaluation_requests_total',
    'Total number of evaluation requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'evaluation_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

EVALUATION_DURATION = Histogram(
    'deepeval_evaluation_duration_seconds',
    'DeepEval evaluation duration in seconds',
    ['model_name', 'metric_type']
)

EVALUATION_SCORES = Histogram(
    'deepeval_scores',
    'DeepEval metric scores',
    ['model_name', 'metric_name']
)

ACTIVE_EVALUATIONS = Gauge(
    'active_evaluations',
    'Number of currently running evaluations'
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)


def setup_metrics(app: FastAPI):
    """Setup Prometheus metrics for FastAPI app."""
    
    # Instrument FastAPI with default metrics
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="fastapi_inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        # Update system metrics
        try:
            SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
            SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
        except Exception as e:
            print(f"Error updating system metrics: {e}")
        
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    return instrumentator


def record_evaluation_metrics(model_name: str, metric_name: str, score: float, duration: float):
    """Record evaluation-specific metrics."""
    EVALUATION_SCORES.labels(model_name=model_name, metric_name=metric_name).observe(score)
    EVALUATION_DURATION.labels(model_name=model_name, metric_type=metric_name).observe(duration)


def increment_active_evaluations():
    """Increment active evaluations counter."""
    ACTIVE_EVALUATIONS.inc()


def decrement_active_evaluations():
    """Decrement active evaluations counter."""
    ACTIVE_EVALUATIONS.dec()