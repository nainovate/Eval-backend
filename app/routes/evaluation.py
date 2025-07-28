"""Updated evaluation API endpoints for task-type based YAML datasets."""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
import structlog
import yaml
from pathlib import Path

from app.models.response import APIResponse
from app.db.connection import get_database
from app.services.evaluation_service import evaluation_service
from app.telemetry.setup import get_tracer

logger = structlog.get_logger()
tracer = get_tracer(__name__)
router = APIRouter(prefix="/api/v1/evaluations", tags=["Evaluations"])


class EvaluationRequest(BaseModel):
    """Request model for dataset evaluation."""
    file_path: str
    metrics: List[str] = Field(
        default=["answer_relevancy", "faithfulness", "bias"],
        description="Direct list of DeepEval metric names"
    )
    metric_category: Optional[str] = Field(
        default=None,
        description="Optional: run all metrics from category (rag-metrics, safety-ethics, task-specific, custom-metrics)"
    )


class DatasetRow(BaseModel):
    """Flexible dataset row model for different task types with model responses."""
    # Question Answering
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    context: Optional[str] = None
    
    # Summarization
    input_text: Optional[str] = None
    expected_summary: Optional[str] = None
    
    # Conversational QA
    conversation_history: Optional[str] = None
    
    # Retrieval (RAG)
    query: Optional[str] = None
    retrieved_documents: Optional[str] = None
    
    # Classification
    expected_label: Optional[str] = None
    
    # Structured Output Generation
    input_instruction: Optional[str] = None
    expected_output: Optional[str] = None
    
    # Open-ended Generation
    prompt: Optional[str] = None
    reference_output: Optional[str] = None
    
    # Model responses - the key addition for your YAML structure
    model_responses: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of model_name: generated_response"
    )
    
    # Common optional fields
    reference: Optional[str] = None
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


class TaskTypeDataset(BaseModel):
    """Dataset model for task-type based YAML structure with model responses."""
    task_type: str
    description: Optional[str] = None
    data: List[DatasetRow] = Field(alias="data")  # Support "data" field name from your YAML
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        allow_population_by_field_name = True


@router.get("/metrics")
async def get_available_metrics():
    """Get list of available DeepEval metrics organized by category."""
    
    with tracer.start_as_current_span("get_available_metrics") as span:
        try:
            metric_info = evaluation_service.get_metric_info()
            
            span.set_attribute("total_metrics", metric_info["total_count"])
            
            logger.info(
                "Retrieved available metrics",
                total_metrics=metric_info["total_count"],
                categories=list(metric_info["metric_categories"].keys())
            )
            
            return {
                "success": True,
                "data": {
                    "metric_categories": metric_info["metric_categories"],
                    "all_available_metrics": metric_info["all_available_metrics"],
                    "total_count": metric_info["total_count"],
                    "note": metric_info["note"],
                    "optional_metrics_status": metric_info["optional_metrics_status"],
                    "usage_examples": {
                        "by_category": {
                            "rag_metrics": ["answer_relevancy", "faithfulness", "contextual_precision"],
                            "safety_metrics": ["bias", "toxicity", "hallucination"],
                            "task_specific": ["summarization"] if metric_info["optional_metrics_status"]["summarization"] else [],
                            "custom": ["g_eval"] if metric_info["optional_metrics_status"]["g_eval"] else []
                        },
                        "mixed_selection": ["answer_relevancy", "bias", "summarization"],
                        "use_category": "Set metric_category to 'rag-metrics' to run all RAG metrics"
                    }
                }
            }
            
        except Exception as e:
            logger.error("Failed to get available metrics", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@router.post("/dataset/evaluate", response_model=APIResponse)
async def evaluate_dataset_structured(
    request: EvaluationRequest,
    http_request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Load task-type YAML dataset with model_responses and evaluate with specified DeepEval metrics."""
    
    with tracer.start_as_current_span("evaluate_dataset_structured") as span:
        try:
            # Load YAML file
            yaml_file = Path(request.file_path)
            if not yaml_file.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # Parse the task-type dataset structure
            try:
                dataset = TaskTypeDataset(**yaml_data)
            except Exception as e:
                logger.error("Failed to parse YAML dataset", error=str(e), yaml_structure=yaml_data)
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid YAML structure. Expected task_type, description, and data fields. Error: {str(e)}"
                )
            
            span.set_attribute("dataset.task_type", dataset.task_type)
            span.set_attribute("dataset.size", len(dataset.data))
            span.set_attribute("selected_metrics", str(request.metrics))
            span.set_attribute("metric_category", request.metric_category or "custom_selection")
            
            logger.info(
                "Starting task-type dataset evaluation with model responses",
                task_type=dataset.task_type,
                total_rows=len(dataset.data),
                selected_metrics=request.metrics,
                metric_category=request.metric_category
            )
            
            # Validate metrics if specific metrics provided
            if request.metrics and not request.metric_category:
                available_metrics = evaluation_service.get_available_metrics()
                invalid_metrics = [m for m in request.metrics if m.lower() not in [am.lower() for am in available_metrics]]
                if invalid_metrics:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Invalid metrics: {invalid_metrics}. Available: {available_metrics}"
                    )
            
            # Extract all model names from the dataset
            all_model_names = set()
            for row in dataset.data:
                all_model_names.update(row.model_responses.keys())
            
            all_model_names = list(all_model_names)
            
            if not all_model_names:
                raise HTTPException(
                    status_code=400,
                    detail="No model_responses found in dataset. Please ensure each row has model_responses field."
                )
            
            logger.info(
                "Found models in dataset",
                models=all_model_names,
                total_models=len(all_model_names)
            )
            
            # Prepare data for batch evaluation
            dataset_rows = []
            model_outputs = {model_name: [] for model_name in all_model_names}
            
            for i, row in enumerate(dataset.data):
                row_dict = row.dict(exclude_none=True)
                # Remove model_responses from row_dict as we'll pass it separately
                row_model_responses = row_dict.pop('model_responses', {})
                dataset_rows.append(row_dict)
                
                # Collect model outputs for this row
                for model_name in all_model_names:
                    # Use empty string if model doesn't have response for this row
                    model_output = row_model_responses.get(model_name, "")
                    model_outputs[model_name].append(model_output)
            
            # Validate all models have responses for all rows
            for model_name, outputs in model_outputs.items():
                if len(outputs) != len(dataset_rows):
                    logger.warning(f"Model '{model_name}' missing some responses, padding with empty strings")
                    # Pad with empty strings if needed
                    while len(outputs) < len(dataset_rows):
                        outputs.append("")
            
            # Use the batch evaluation method
            evaluation_result = await evaluation_service.evaluate_dataset_batch_multiple_models(
                dataset_rows=dataset_rows,
                model_outputs=model_outputs,
                task_type=dataset.task_type,
                selected_metrics=request.metrics if request.metrics else None,
                metric_category=request.metric_category
            )
            
            # Extract detailed results for response
            detailed_results = []
            for i, row_result in enumerate(evaluation_result.get("row_results", [])):
                row_data = dataset_rows[i] if i < len(dataset_rows) else {}
                model_scores = row_result.get("model_scores", {})
                
                # Calculate best model for this row
                best_model_for_row = None
                best_score_for_row = -1
                for model_name, scores in model_scores.items():
                    if scores:
                        avg_score = sum(scores.values()) / len(scores)
                        if avg_score > best_score_for_row:
                            best_score_for_row = avg_score
                            best_model_for_row = model_name
                
                detailed_result = {
                    "row_id": i + 1,
                    "input_data": _extract_input_data(row_data, dataset.task_type),
                    "expected_output": _extract_expected_output(row_data, dataset.task_type),
                    "model_evaluations": {},
                    "best_model_for_row": best_model_for_row,
                    "best_score_for_row": round(best_score_for_row, 3) if best_score_for_row > -1 else 0,
                    "category": row_data.get('category', 'general'),
                    "difficulty": row_data.get('difficulty', 'medium')
                }
                
                # Add individual model evaluations
                for model_name, scores in model_scores.items():
                    model_output = model_outputs[model_name][i] if i < len(model_outputs[model_name]) else ""
                    overall_score = sum(scores.values()) / len(scores) if scores else 0
                    
                    detailed_result["model_evaluations"][model_name] = {
                        "model_output": model_output[:200] + "..." if len(model_output) > 200 else model_output,
                        "metric_scores": {k: round(v, 3) for k, v in scores.items()},
                        "overall_score": round(overall_score, 3)
                    }
                
                detailed_results.append(detailed_result)
            
            # Create enhanced summary
            batch_stats = evaluation_result.get("batch_statistics", {})
            model_stats = batch_stats.get("model_statistics", {})
            
            summary = {
                "dataset_info": {
                    "task_type": dataset.task_type,
                    "description": dataset.description,
                    "total_rows": len(dataset.data),
                    "successfully_evaluated": evaluation_result.get("successful_rows", 0),
                    "failed_evaluations": len(evaluation_result.get("failed_rows", []))
                },
                "models_info": {
                    "models_evaluated": all_model_names,
                    "total_models": len(all_model_names),
                    "best_model_overall": batch_stats.get("best_model_overall"),
                    "best_score_overall": round(batch_stats.get("best_score_overall", 0), 3)
                },
                "evaluation_config": {
                    "selected_metrics": request.metrics,
                    "metric_category": request.metric_category,
                    "total_metrics_used": len(request.metrics) if request.metrics else 0
                },
                "model_performance": {}
            }
            
            # Add model performance summary
            for model_name, stats in model_stats.items():
                summary["model_performance"][model_name] = {
                    "overall_average": round(stats.get("overall_average", 0), 3),
                    "metrics_evaluated": stats.get("metrics_evaluated", 0),
                    "metric_averages": {
                        metric: round(metric_stats.get("average", 0), 3)
                        for metric, metric_stats in stats.get("metric_stats", {}).items()
                    }
                }
            
            logger.info(
                "Task-type dataset evaluation with model responses completed",
                task_type=dataset.task_type,
                total_evaluations=len(detailed_results),
                models_evaluated=len(all_model_names),
                best_model_overall=batch_stats.get("best_model_overall"),
                metrics_used=request.metrics or "category_based"
            )
            
            return APIResponse(
                success=True,
                message=f"Dataset evaluation completed. {len(detailed_results)} rows processed across {len(all_model_names)} models.",
                data={
                    "summary": summary,
                    "detailed_results": detailed_results,
                    "full_batch_statistics": batch_stats
                },
                trace_id=getattr(http_request.state, 'trace_id', None)
            )
            
        except Exception as e:
            logger.error("Failed to evaluate task-type dataset with model responses", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

# Add the helper methods as module-level functions
def _extract_input_data(row_dict: Dict[str, Any], task_type: str) -> str:
    """Extract the input data based on task type."""
    if task_type == 'Question Answering':
        return row_dict.get('question', '')
    elif task_type == 'Summarization':
        return row_dict.get('input_text', '')
    elif task_type == 'Conversational QA':
        history = row_dict.get('conversation_history', '')
        question = row_dict.get('question', '')
        return f"{history}\nUser: {question}" if history else question
    elif task_type == 'Retrieval (RAG)':
        return row_dict.get('query', '')
    elif task_type == 'Classification':
        return row_dict.get('input_text', '')
    elif task_type == 'Structured Output Generation':
        return row_dict.get('input_instruction', '')
    elif task_type == 'Open-ended Generation':
        return row_dict.get('prompt', '')
    else:
        # Fallback: try to find any input-like field
        for key in ['question', 'input_text', 'prompt', 'query', 'input_instruction']:
            if key in row_dict:
                return row_dict[key]
        return str(row_dict)

def _extract_expected_output(row_dict: Dict[str, Any], task_type: str) -> str:
    """Extract the expected output based on task type."""
    if task_type == 'Question Answering':
        return row_dict.get('expected_answer', '')
    elif task_type == 'Summarization':
        return row_dict.get('expected_summary', '')
    elif task_type == 'Conversational QA':
        return row_dict.get('expected_answer', '')
    elif task_type == 'Retrieval (RAG)':
        return row_dict.get('expected_answer', '')
    elif task_type == 'Classification':
        return row_dict.get('expected_label', '')
    elif task_type == 'Structured Output Generation':
        return row_dict.get('expected_output', '')
    elif task_type == 'Open-ended Generation':
        return row_dict.get('reference_output', '')
    else:
        # Fallback: try to find any expected-like field
        for key in ['expected_answer', 'expected_summary', 'expected_output', 'expected_label']:
            if key in row_dict:
                return row_dict[key]
        return ""


