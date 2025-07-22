from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    trace_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: str
    version: str
    dependencies: Dict[str, str]