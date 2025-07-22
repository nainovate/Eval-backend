"""Evaluation data models."""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId for Pydantic v2."""
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.no_info_plain_validator_function(cls.validate)
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError("Invalid ObjectId")
    
    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, handler):
        field_schema.update(type="string", format="objectid")

class EvaluationRequest(BaseModel):
    """Model for evaluation request."""
    model_name: str = Field(..., description="Name of the model being evaluated")
    prompt: str = Field(..., description="Input prompt for evaluation")
    response: str = Field(..., description="Model's response to evaluate")
    expected_output: Optional[str] = Field(None, description="Expected output for comparison")
    evaluation_type: str = Field("general", description="Type of evaluation to perform")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Model for evaluation results."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    model_name: str
    prompt: str
    response: str
    expected_output: Optional[str] = None
    
    # DeepEval metrics
    accuracy_score: Optional[float] = None
    relevancy_score: Optional[float] = None
    coherence_score: Optional[float] = None
    fluency_score: Optional[float] = None
    overall_score: Optional[float] = None
    
    # Metadata
    evaluation_type: str = "general"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evaluation_duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ComparisonRequest(BaseModel):
    """Model for comparing multiple model evaluations."""
    evaluation_ids: List[str] = Field(..., description="List of evaluation IDs to compare")
    comparison_metrics: List[str] = Field(
        default=["accuracy_score", "relevancy_score", "coherence_score"],
        description="Metrics to compare"
    )


class ComparisonResult(BaseModel):
    """Model for comparison results."""
    comparison_id: str = Field(default_factory=lambda: str(ObjectId()))
    evaluations: List[EvaluationResult]
    summary: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
class BatchEvaluationRequest(BaseModel):
    """Model for batch evaluation request with multiple models."""
    input: str = Field(..., description="Input prompt for evaluation")
    expected_output: str = Field(..., description="Expected output for comparison")
    model_responses: Dict[str, str] = Field(..., description="Responses from different models")
    context: Optional[str] = Field(None, description="Additional context")
    category: Optional[str] = Field(None, description="Category of the evaluation")
    difficulty: Optional[str] = Field("medium", description="Difficulty level")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DatasetEvaluation(BaseModel):
    """Model for a single evaluation item in a dataset."""
    input: str
    expected_output: str
    model_responses: Dict[str, str]
    context: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = "medium"


class EvaluationDataset(BaseModel):
    """Model for evaluation dataset."""
    dataset_name: str
    description: Optional[str] = None
    task_type: str = "question_answering"
    version: str = "1.0"
    evaluations: List[DatasetEvaluation]


class BatchEvaluationResult(BaseModel):
    """Model for batch evaluation results."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    dataset_name: str
    input: str
    expected_output: str
    context: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    
    # Model results
    model_results: Dict[str, EvaluationResult] = Field(default_factory=dict)
    
    # Comparative metrics
    best_model: Optional[str] = None
    worst_model: Optional[str] = None
    average_score: Optional[float] = None
    score_variance: Optional[float] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evaluation_duration_ms: Optional[float] = None
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}