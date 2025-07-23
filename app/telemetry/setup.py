"""OpenTelemetry configuration and setup - Fixed service name propagation."""
import logging
import os
import atexit
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.semconv.resource import ResourceAttributes
import structlog

from app.config import settings

# Global references for cleanup and preventing multiple initialization
_span_processors = []
_tracer_provider = None
_is_initialized = False


def setup_telemetry():
    """Initialize OpenTelemetry with proper service name propagation."""
    global _tracer_provider, _is_initialized
    
    # Prevent multiple initialization
    if _is_initialized:
        print("⚠️ Telemetry already initialized, skipping...")
        return
    
    print(f"🚀 Setting up telemetry for service: {settings.otel_service_name}")
    
    # Create resource with proper service identification using semantic conventions
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: settings.otel_service_name,
        ResourceAttributes.SERVICE_VERSION: settings.otel_service_version,
        ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", "localhost"),
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
        # Additional attributes for better identification
        "service.namespace": "evaluation",
        "telemetry.setup.version": "fixed",
    })
    
    print(f"📋 Resource attributes: {dict(resource.attributes)}")
    
    # Setup tracing with proper service name
    _tracer_provider = setup_tracing(resource)
    
    # Setup metrics
    setup_metrics(resource)
    
    # Setup logging
    setup_logging()
    
    # Register cleanup on exit
    atexit.register(cleanup_telemetry)
    
    _is_initialized = True
    print(f"✅ Telemetry initialized successfully for: {settings.otel_service_name}")


def setup_tracing(resource: Resource):
    """Configure distributed tracing with proper service name."""
    global _span_processors
    
    try:
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry import trace

        # Create new tracer provider with our resource (don't reuse existing)
        tracer_provider = TracerProvider(resource=resource)
        
        # Configure Jaeger endpoint
        jaeger_endpoint = settings.jaeger_endpoint
        print(f"🔍 Configuring Jaeger exporter: {jaeger_endpoint}")
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            collector_endpoint=jaeger_endpoint,
        )

        # Create batch span processor with shorter delays for debugging
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout_millis=30000,  # 30 seconds
            schedule_delay_millis=500,    # 0.5 seconds
        )
        
        tracer_provider.add_span_processor(span_processor)
        _span_processors.append(span_processor)

        # Set the tracer provider globally BEFORE any instrumentation
        trace.set_tracer_provider(tracer_provider)
        
        print("✅ Jaeger HTTP tracing configured successfully")
        
        # Test trace with service name verification
        test_initial_trace()
        
        return tracer_provider

    except Exception as e:
        print(f"❌ Tracing setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_initial_trace():
    """Generate a test trace to verify service name propagation."""
    try:
        tracer = trace.get_tracer("telemetry-setup-test")
        with tracer.start_as_current_span("telemetry_initialization") as span:
            span.set_attribute("test.purpose", "verify_service_name")
            span.set_attribute("service.name", settings.otel_service_name)
            span.set_attribute("setup.phase", "initialization")
            span.add_event("Telemetry setup completed")
            
            # Get span details for logging
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            print(f"🧪 Test trace generated: {trace_id}")
            
            # Verify the resource service name is properly set
            current_provider = trace.get_tracer_provider()
            if hasattr(current_provider, 'resource'):
                service_name = current_provider.resource.attributes.get(ResourceAttributes.SERVICE_NAME, "UNKNOWN")
                print(f"🔍 TracerProvider service name: {service_name}")
                
                if service_name == settings.otel_service_name:
                    print("✅ Service name correctly propagated to TracerProvider")
                else:
                    print(f"❌ Service name mismatch! Expected: {settings.otel_service_name}, Got: {service_name}")
            else:
                print("⚠️ TracerProvider has no resource attribute")
            
    except Exception as e:
        print(f"⚠️ Test trace generation failed: {e}")


def setup_auto_instrumentation(app=None):
    """Setup automatic instrumentation after telemetry is initialized."""
    if not _is_initialized:
        print("⚠️ Telemetry not initialized, skipping auto-instrumentation")
        return
    
    try:
        # Instrument FastAPI with the app instance
        if app:
            # Check if already instrumented to avoid double instrumentation
            if not getattr(app, '_is_instrumented_by_opentelemetry', False):
                FastAPIInstrumentor.instrument_app(app)
                print("✅ FastAPI app instrumentation enabled")
            else:
                print("ℹ️ FastAPI app already instrumented")
        
        # Global FastAPI instrumentation as fallback
        try:
            instrumentor = FastAPIInstrumentor()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
                print("✅ FastAPI global instrumentation enabled")
            else:
                print("ℹ️ FastAPI globally already instrumented")
        except Exception as e:
            print(f"⚠️ FastAPI global instrumentation failed: {e}")
        
    except Exception as e:
        print(f"⚠️ FastAPI instrumentation failed: {e}")
    
    try:
        # Instrument HTTP requests
        requests_instrumentor = RequestsInstrumentor()
        if not requests_instrumentor.is_instrumented_by_opentelemetry:
            requests_instrumentor.instrument()
            print("✅ Requests instrumentation enabled")
        else:
            print("ℹ️ Requests already instrumented")
        
    except Exception as e:
        print(f"⚠️ Requests instrumentation failed: {e}")


def setup_metrics(resource: Resource):
    """Configure metrics with Prometheus."""
    try:
        from prometheus_client import start_http_server
        
        # Start Prometheus metrics server
        try:
            start_http_server(settings.prometheus_port)
            print(f"✅ Prometheus metrics server started on port {settings.prometheus_port}")
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"⚠️ Prometheus port {settings.prometheus_port} already in use")
            else:
                raise
        
        meter_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(meter_provider)
        
        print(f"✅ Prometheus metrics configured")
        
    except Exception as e:
        print(f"⚠️ Metrics setup failed: {e}")


def setup_logging():
    """Configure structured logging with file output."""
    try:
        # Ensure log directory exists
        log_dir = os.path.dirname(settings.log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup file logging
        logging.basicConfig(
            format="%(message)s",
            handlers=[
                logging.FileHandler(settings.log_file_path),
                logging.StreamHandler()  # Also log to console
            ],
            level=getattr(logging, settings.log_level.upper()),
        )
        
        print("✅ Structured logging configured")
        
    except Exception as e:
        print(f"⚠️ Logging setup failed: {e}")


def cleanup_telemetry():
    """Clean up telemetry resources and force final export."""
    global _span_processors
    
    print("🧹 Cleaning up telemetry...")
    
    for processor in _span_processors:
        try:
            print("🚀 Forcing final span export...")
            processor.force_flush(timeout_millis=10000)
            processor.shutdown()
        except Exception as e:
            print(f"⚠️ Error during span processor cleanup: {e}")
    
    print("✅ Telemetry cleanup completed")


def get_tracer(name: str):
    """Get a tracer instance for manual span creation."""
    return trace.get_tracer(name, settings.otel_service_version)


def force_export_spans():
    """Force export all pending spans (useful for debugging)."""
    global _span_processors
    
    for processor in _span_processors:
        try:
            processor.force_flush(timeout_millis=5000)
            print("✅ Forced span export completed")
        except Exception as e:
            print(f"⚠️ Force export failed: {e}")


def verify_service_name():
    """Verify that the service name is properly configured."""
    try:
        current_provider = trace.get_tracer_provider()
        if hasattr(current_provider, 'resource'):
            service_name = current_provider.resource.attributes.get(ResourceAttributes.SERVICE_NAME, "UNKNOWN")
            print(f"🔍 Current service name in TracerProvider: {service_name}")
            return service_name
        else:
            print("⚠️ TracerProvider has no resource attribute")
            return "UNKNOWN"
    except Exception as e:
        print(f"⚠️ Error verifying service name: {e}")
        return "ERROR"