"""FastAPI application entry point with minimal DeepEval metrics support."""
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import settings
from app.db.connection import db_manager
from app.telemetry.setup import setup_telemetry, setup_auto_instrumentation
from app.telemetry.metrics import setup_metrics
from app.telemetry.middleware import TelemetryMiddleware, SecurityMiddleware
from app.routes import health
from app.routes.evaluation import router as evaluation_router
from app.services.evaluation_service import evaluation_service

# Initialize telemetry BEFORE creating FastAPI app
print("🚀 Initializing telemetry...")
setup_telemetry()

# Verify service name is set correctly
from app.telemetry.setup import verify_service_name
verify_service_name()

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Claude Evaluation Backend", 
                version=settings.app_version,
                service_name=settings.otel_service_name)
    
    # Log minimal DeepEval metrics info
    #metrics_info = evaluation_service.get_metric_info()
    
    
    try:
        await db_manager.connect()
        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down application")
        await db_manager.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-ready FastAPI backend for Claude AI evaluation with core DeepEval metrics and full observability",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Setup automatic instrumentation AFTER app creation
    setup_auto_instrumentation(app)
    
    # Setup metrics (before middleware)
    setup_metrics(app)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware (TelemetryMiddleware should be LAST)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(TelemetryMiddleware)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(evaluation_router)
    
    @app.get("/")
    async def root():
        """Root endpoint with minimal DeepEval metrics info."""
        from opentelemetry import trace
        
        # Add manual span for testing
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("root_endpoint") as span:
            span.set_attribute("endpoint", "/")
            span.set_attribute("service.name", settings.otel_service_name)
            span.add_event("Root endpoint accessed")
            
            # Get minimal DeepEval metrics info
            metrics_info = evaluation_service.get_metric_info()
            
            return {
                "service": settings.app_name,
                "version": settings.app_version,
                "status": "running",
                "docs_url": "/docs" if settings.debug else "disabled",
                "metrics_url": "/metrics",
                "jaeger_endpoint": settings.jaeger_endpoint,
                "service_name": settings.otel_service_name,
                "deepeval_metrics": {
                    "available_metrics": metrics_info["available_metrics"],
                    "total_count": metrics_info["total_count"],
                    "optional_metrics_status": metrics_info["optional_metrics_status"],
                    "metrics_endpoint": "/api/v1/evaluations/metrics",
                    "evaluation_endpoint": "/api/v1/evaluations/dataset/evaluate",
                    "batch_endpoint": "/api/v1/evaluations/batch"
                }
            }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    # Print configuration info
    print(f"🔧 Service Name: {settings.otel_service_name}")
    print(f"🔧 Jaeger Endpoint: {settings.jaeger_endpoint}")
    print(f"🔧 Prometheus Port: {settings.prometheus_port}")
    
   
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=False  # We handle logging in middleware
    )