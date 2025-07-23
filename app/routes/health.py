"""Health check endpoints with improved Jaeger testing."""
from datetime import datetime
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
import structlog
import time

from app.config import settings
from app.models.response import HealthResponse
from app.db.connection import get_database
from app.telemetry.setup import get_tracer, force_export_spans

logger = structlog.get_logger()
router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Basic health check endpoint."""
    
    dependencies = {"database": "unknown"}
    
    # Check MongoDB connection
    try:
        await db.command("ping")
        dependencies["database"] = "healthy"
    except Exception as e:
        logger.warning("Database health check failed", error=str(e))
        dependencies["database"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if dependencies["database"] == "healthy" else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version=settings.app_version,
        dependencies=dependencies
    )


@router.get("/ready")
async def readiness_check(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Readiness check for Kubernetes."""
    try:
        await db.command("ping")
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}


@router.get("/test-jaeger")
async def test_jaeger_trace():
    """Test endpoint to verify Jaeger trace generation with forced export."""
    
    tracer = get_tracer("health-test-tracer")
    
    try:
        with tracer.start_as_current_span("test_jaeger_span") as span:
            # Set comprehensive attributes
            span.set_attribute("test.key", "test_value")
            span.set_attribute("service.name", settings.otel_service_name)
            span.set_attribute("test.endpoint", "/health/test-jaeger")
            span.set_attribute("test.timestamp", time.time())
            span.set_attribute("environment", "debug")
            
            # Add events
            span.add_event("Test event for Jaeger verification")
            span.add_event("Service configuration", {
                "service_name": settings.otel_service_name,
                "jaeger_endpoint": settings.jaeger_endpoint
            })
            
            # Get span details
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            print(f"🔍 Generated trace: {trace_id}")
            print(f"🔍 Generated span: {span_id}")
            print(f"🔍 Service name: {settings.otel_service_name}")
            print(f"🔍 Jaeger endpoint: {settings.jaeger_endpoint}")
            
            # Simulate some work
            time.sleep(0.1)
            span.add_event("Test work completed")
            
            # Force export before returning
            print("🚀 Forcing span export...")
            force_export_spans()
            
            return {
                "message": "Test trace generated and exported",
                "trace_id": trace_id,
                "span_id": span_id,
                "service": settings.otel_service_name,
                "jaeger_endpoint": settings.jaeger_endpoint,
                "jaeger_ui": "http://localhost:16686",
                "search_url": f"http://localhost:16686/trace/{trace_id}",
                "instructions": [
                    "1. Wait 10-30 seconds for trace to appear",
                    "2. Go to Jaeger UI: http://localhost:16686",
                    f"3. Look for service: {settings.otel_service_name}",
                    f"4. Search for trace ID: {trace_id}"
                ]
            }
    
    except Exception as e:
        print(f"❌ Error creating span: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


@router.get("/test-jaeger-advanced")
async def test_jaeger_advanced():
    """Advanced Jaeger test with multiple spans and detailed logging."""
    
    tracer = get_tracer("health-advanced-test")
    
    try:
        with tracer.start_as_current_span("advanced_test_parent") as parent_span:
            parent_span.set_attribute("test.type", "advanced_jaeger_test")
            parent_span.set_attribute("service.name", settings.otel_service_name)
            parent_span.add_event("Starting advanced test")
            
            # Create child spans
            with tracer.start_as_current_span("child_span_1") as child1:
                child1.set_attribute("child.number", 1)
                child1.add_event("Child span 1 work")
                time.sleep(0.05)
            
            with tracer.start_as_current_span("child_span_2") as child2:
                child2.set_attribute("child.number", 2)
                child2.add_event("Child span 2 work")
                time.sleep(0.05)
            
            # Get parent span details
            span_context = parent_span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            parent_span.add_event("Advanced test completed")
            
            print(f"🔍 Advanced test trace: {trace_id}")
            print(f"🔍 Parent span: {span_id}")
            
            # Force export
            print("🚀 Forcing advanced test export...")
            force_export_spans()
            
            return {
                "message": "Advanced test trace generated",
                "trace_id": trace_id,
                "parent_span_id": span_id,
                "service": settings.otel_service_name,
                "spans_created": 3,
                "search_url": f"http://localhost:16686/trace/{trace_id}"
            }
    
    except Exception as e:
        print(f"❌ Advanced test error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}