"""OpenTelemetry configuration and setup."""
import logging
import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from prometheus_client import start_http_server
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import structlog

from app.config import settings


def setup_telemetry():
    """Initialize OpenTelemetry with Jaeger, Prometheus, and structured logging."""
    
    # Create resource
    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": settings.otel_service_version,
        "service.instance.id": os.getenv("HOSTNAME", "localhost"),
    })
    
    # Setup tracing
    setup_tracing(resource)
    
    # Setup metrics
    setup_metrics(resource)
    
    # Setup logging
    setup_logging()


def setup_tracing(resource: Resource):
    """Configure distributed tracing with Jaeger via HTTP collector."""
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry import trace

        tracer_provider = TracerProvider(resource=resource)

        # Use Jaeger HTTP collector (port 14268)
        jaeger_exporter = JaegerExporter(
            collector_endpoint="http://localhost:14268/api/traces",
        )

        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)

        trace.set_tracer_provider(tracer_provider)

        print("✅ Jaeger HTTP tracing configured (port 14268)")

    except Exception as e:
        print(f"❌ Tracing setup failed: {e}")


def setup_metrics(resource: Resource):
    """Configure metrics with Prometheus."""
    try:
        start_http_server(settings.prometheus_port)
        
        metric_reader = None
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            metric_reader = PrometheusMetricReader()
        except ImportError:
            print("⚠️ Prometheus metric reader not available")
        
        if metric_reader:
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
        else:
            meter_provider = MeterProvider(resource=resource)
        
        metrics.set_meter_provider(meter_provider)
        print(f"✅ Prometheus metrics configured on port {settings.prometheus_port}")
        
    except Exception as e:
        print(f"⚠️ Metrics setup failed: {e}")


def setup_logging():
    """Configure structured logging with file output."""
    # Ensure log directory exists
    os.makedirs(os.path.dirname(settings.log_file_path), exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(
            file=open(settings.log_file_path, "a")
        ),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_tracer(name: str):
    """Get a tracer instance."""
    return trace.get_tracer(name)


def get_meter(name: str):
    """Get a meter instance."""
    return metrics.get_meter(name)