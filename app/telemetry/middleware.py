"""Custom middleware for telemetry and logging."""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog
from opentelemetry import trace

from app.models.response import ErrorResponse

logger = structlog.get_logger()


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and tracing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with telemetry."""
        # Generate trace ID if not present
        trace_id = str(uuid.uuid4())
        
        # Get current span
        span = trace.get_current_span()
        if span:
            span_context = span.get_span_context()
            if span_context:
                trace_id = format(span_context.trace_id, '032x')
        
        # Add trace ID to request state
        request.state.trace_id = trace_id
        
        # Log request
        start_time = time.time()
        
        with structlog.contextvars.bound_contextvars(
            trace_id=trace_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent")
        ):
            logger.info("Request started")
            
            try:
                response = await call_next(request)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Log response
                logger.info(
                    "Request completed",
                    status_code=response.status_code,
                    duration_ms=round(duration_ms, 2)
                )
                
                # Add trace ID to response headers
                response.headers["X-Trace-ID"] = trace_id
                
                return response
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Request failed",
                    error=str(e),
                    duration_ms=round(duration_ms, 2),
                    exc_info=True
                )
                
                # Return structured error response
                error_response = ErrorResponse(
                    message="Internal server error",
                    error_code="INTERNAL_ERROR",
                    trace_id=trace_id
                )
                
                return JSONResponse(
                    status_code=500,
                    content=error_response.dict(),
                    headers={"X-Trace-ID": trace_id}
                )


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and CORS."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response