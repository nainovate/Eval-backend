"""Simple evaluation API endpoints - fixed with proper test_id handling."""
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field
import structlog
import yaml
from pathlib import Path

from app.models.evaluation import (
    BatchEvaluationRequest,
    BatchEvaluationResult,
    EvaluationDataset
)
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


@router.get("/metrics")
async def get_available_metrics():
    """Get list of available DeepEval metrics."""
    
    with tracer.start_as_current_span("get_available_metrics") as span:
        try:
            metric_info = evaluation_service.get_metric_info()
            
            span.set_attribute("total_metrics", metric_info["total_count"])
            span.set_attribute("optional_metrics_status", str(metric_info["optional_metrics_status"]))
            
            logger.info(
                "Retrieved available metrics",
                total_metrics=metric_info["total_count"],
                optional_metrics_status=metric_info["optional_metrics_status"]
            )
            
            return {
                "success": True,
                "data": {
                    "available_metrics": metric_info["available_metrics"],
                    "total_count": metric_info["total_count"],
                    "optional_metrics_status": metric_info["optional_metrics_status"],
                    "metric_details": metric_info["metric_details"],
                    "usage_example": {
                        "basic": ["answer_relevancy", "faithfulness", "bias"],
                        "comprehensive": ["answer_relevancy", "faithfulness", "contextual_precision", "bias", "toxicity"],
                        "with_geval": ["answer_relevancy", "faithfulness", "g_eval"] if metric_info["optional_metrics_status"]["g_eval"] else "g_eval not available"
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
    """Load YAML dataset and evaluate with specified DeepEval metrics."""
    
    with tracer.start_as_current_span("evaluate_dataset_structured") as span:
        try:
            # Load YAML file
            yaml_file = Path(request.file_path)
            if not yaml_file.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            
            with open(yaml_file, 'r', encoding='utf-8') as f:
                dataset_data = yaml.safe_load(f)
            
            dataset = EvaluationDataset(**dataset_data)
            
            span.set_attribute("dataset.name", dataset.dataset_name)
            span.set_attribute("dataset.size", len(dataset.evaluations))
            span.set_attribute("selected_metrics", str(request.metrics))
            
            logger.info(
                "Starting dataset evaluation",
                dataset_name=dataset.dataset_name,
                total_evaluations=len(dataset.evaluations),
                selected_metrics=request.metrics
            )
            
            # Validate metrics
            available_metrics = evaluation_service.get_available_metrics()
            invalid_metrics = [m for m in request.metrics if m.lower() not in [am.lower() for am in available_metrics]]
            if invalid_metrics:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid metrics: {invalid_metrics}. Available: {available_metrics}"
                )
            
            # Process each test case
            test_results = []
            all_model_scores = {}
            metric_aggregated_data = {metric.lower(): {} for metric in request.metrics}
            
            for i, eval_item in enumerate(dataset.evaluations):
                # Create batch evaluation request
                batch_request = BatchEvaluationRequest(
                    input=eval_item.input,
                    expected_output=eval_item.expected_output,
                    model_responses=eval_item.model_responses,
                    context=eval_item.context,
                    category=eval_item.category,
                    difficulty=eval_item.difficulty
                )
                
                # Evaluate with specified metrics
                batch_result = await evaluation_service.evaluate_batch(
                    batch_request, 
                    request.metrics
                )
                
                # Store in database
                collection = db.get_collection("evaluations")
                inserted = await collection.insert_one(batch_result.dict(by_alias=True))
                batch_result.id = inserted.inserted_id
                
                # Extract results for each model
                responses = []
                
                for model_name, model_result in batch_result.model_results.items():
                    # Get DeepEval scores
                    deepeval_scores = model_result.metadata.get("deepeval_scores", {})
                    
                    # Process metric scores
                    metrics = {}
                    metric_scores = []
                    
                    for metric_name, score in deepeval_scores.items():
                        score = round(score, 2)
                        metrics[metric_name] = score
                        metric_scores.append(score)
                        
                        # Aggregate for summary
                        if metric_name in metric_aggregated_data:
                            if model_name not in metric_aggregated_data[metric_name]:
                                metric_aggregated_data[metric_name][model_name] = []
                            metric_aggregated_data[metric_name][model_name].append(score)
                    
                    # Calculate overall score
                    overall_score = round(sum(metric_scores) / len(metric_scores), 2) if metric_scores else 0.0
                    
                    # Track for model ranking
                    if model_name not in all_model_scores:
                        all_model_scores[model_name] = []
                    all_model_scores[model_name].append(overall_score)
                    
                    responses.append({
                        "model": model_name,
                        "overall_score": overall_score,
                        "metrics": metrics,
                        "evaluation_duration_ms": model_result.evaluation_duration_ms
                    })
                
                # Fixed: Generate test_id instead of accessing eval_item.test_id
                test_id = getattr(eval_item, 'test_id', f"test_{i+1:03d}")
                
                test_results.append({
                    "test_id": test_id,
                    "input": eval_item.input[:100] + "..." if len(eval_item.input) > 100 else eval_item.input,
                    "expected_output": eval_item.expected_output,
                    "category": eval_item.category,
                    "difficulty": eval_item.difficulty,
                    "responses": responses,
                    "evaluation_id": str(batch_result.id)
                })
                
                logger.info(
                    f"Completed evaluation {i+1}/{len(dataset.evaluations)}",
                    test_id=test_id,
                    evaluation_id=str(batch_result.id)
                )
            
            # Calculate model rankings
            model_average_scores = []
            for model_name, scores_list in all_model_scores.items():
                avg_score = round(sum(scores_list) / len(scores_list), 2)
                model_average_scores.append({
                    "model": model_name,
                    "average_score": avg_score
                })
            
            # Sort by score descending
            model_average_scores.sort(key=lambda x: x["average_score"], reverse=True)
            
            # Find best model
            overall_best_model = model_average_scores[0]["model"] if model_average_scores else None
            
            # Calculate metric aggregations
            metrics_aggregated = []
            for metric_name, model_scores in metric_aggregated_data.items():
                if model_scores:  # Only include metrics with data
                    avg_scores = {}
                    for model_name, scores_list in model_scores.items():
                        avg_scores[model_name] = round(sum(scores_list) / len(scores_list), 2)
                    
                    metrics_aggregated.append({
                        "metric": metric_name,
                        "scores": avg_scores
                    })
            
            # Create summary
            summary = {
                "total_test_cases": len(test_results),
                "models_evaluated": list(all_model_scores.keys()),
                "overall_best_model": overall_best_model,
                "model_average_scores": model_average_scores,
                "deepeval_metrics_used": request.metrics,
                "total_metrics_evaluated": len(metrics_aggregated)
            }
            
            logger.info(
                "Dataset evaluation completed",
                dataset_name=dataset.dataset_name,
                total_evaluations=len(test_results),
                best_model=overall_best_model,
                metrics_used=request.metrics
            )
            
            return APIResponse(
                success=True,
                message=f"Evaluation completed successfully. {len(test_results)} test cases processed with {len(request.metrics)} DeepEval metrics.",
                data={
                    "summary": summary,
                    "metrics_aggregated": metrics_aggregated,
                    "test_results": test_results
                },
                trace_id=getattr(http_request.state, 'trace_id', None)
            )
            
        except Exception as e:
            logger.error("Failed to evaluate dataset", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/batch", response_model=APIResponse)
async def create_batch_evaluation(
    request: BatchEvaluationRequest,
    metrics: List[str],
    http_request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Create a batch evaluation for multiple models with specified metrics."""
    
    with tracer.start_as_current_span("batch_evaluation") as span:
        try:
            span.set_attribute("batch.models", str(list(request.model_responses.keys())))
            span.set_attribute("batch.metrics", str(metrics))
            
            logger.info(
                "Starting batch evaluation",
                models=list(request.model_responses.keys()),
                selected_metrics=metrics
            )
            
            # Validate metrics
            available_metrics = evaluation_service.get_available_metrics()
            invalid_metrics = [m for m in metrics if m.lower() not in [am.lower() for am in available_metrics]]
            if invalid_metrics:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid metrics: {invalid_metrics}. Available: {available_metrics}"
                )
            
            # Run evaluation
            result = await evaluation_service.evaluate_batch(request, metrics)
            
            # Store in database
            collection = db.get_collection("batch_evaluations")
            inserted = await collection.insert_one(result.dict(by_alias=True))
            result.id = inserted.inserted_id
            
            logger.info(
                "Batch evaluation completed",
                evaluation_id=str(result.id),
                best_model=result.best_model,
                models_count=len(result.model_results)
            )
            
            return APIResponse(
                success=True,
                message="Batch evaluation completed successfully",
                data={
                    "evaluation_id": str(result.id),
                    "best_model": result.best_model,
                    "model_results": {
                        name: {
                            "overall_score": res.overall_score,
                            "deepeval_scores": res.metadata.get("deepeval_scores", {}),
                            "duration_ms": res.evaluation_duration_ms
                        } 
                        for name, res in result.model_results.items()
                    },
                    "evaluation_duration_ms": result.evaluation_duration_ms
                },
                trace_id=getattr(http_request.state, 'trace_id', None)
            )
            
        except Exception as e:
            logger.error("Batch evaluation failed", error=str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")