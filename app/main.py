"""FastAPI application entry point."""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import settings
from app.db.connection import db_manager
from app.telemetry.setup import setup_telemetry
from app.telemetry.metrics import setup_metrics  # Add this import
from app.telemetry.middleware import TelemetryMiddleware, SecurityMiddleware
from app.routes import health, evaluation

# Setup telemetry before anything else
setup_telemetry()

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Claude Evaluation Backend", version=settings.app_version)
    
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
        description="Production-ready FastAPI backend for Claude AI evaluation with full observability",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # Setup metrics FIRST (before middleware)
    setup_metrics(app)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(TelemetryMiddleware)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(evaluation.router)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs_url": "/docs" if settings.debug else "disabled",
            "metrics_url": "/metrics"  # Add this
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=False  # We handle logging in middleware
    )