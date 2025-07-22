"""Health check endpoints."""
from datetime import datetime
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
import structlog

from app.config import settings
from app.models.response import HealthResponse
from app.db.connection import get_database

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

from opentelemetry import trace

@router.get("/test-jaeger")
async def test_jaeger_trace():
    """Test endpoint to verify Jaeger trace generation."""
    
    tracer = trace.get_tracer("test-tracer")
    
    try:
        with tracer.start_as_current_span("test_jaeger_span") as span:
            span.set_attribute("test.key", "test_value")
            span.set_attribute("service.name", "claude-evaluation-service")
            span.add_event("Test event for Jaeger verification")
            
            # Log span details
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            print(f"🔍 Generated trace: {trace_id}")
            print(f"🔍 Generated span: {span_id}")
            
            return {
                "message": "Test trace generated",
                "trace_id": trace_id,
                "span_id": span_id,
                "service": "claude-evaluation-service"
            }
    
    except Exception as e:
        print(f"❌ Error creating span: {e}")
        return {"error": str(e)}