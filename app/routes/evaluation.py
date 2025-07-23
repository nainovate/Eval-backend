"""Evaluation API endpoints - DeepEval metrics only."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel
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
    metrics: List[str]


@router.post("/dataset/evaluate", response_model=APIResponse)
async def evaluate_dataset_structured(
    request: EvaluationRequest,
    http_request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Load YAML dataset and evaluate with DeepEval metrics."""
    
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
            
            # Process each test case
            test_results = []
            all_model_scores = {}
            metric_aggregated_data = {metric: {} for metric in request.metrics}
            
            for i, eval_item in enumerate(dataset.evaluations):
                # Run evaluation with selected DeepEval metrics
                batch_request = BatchEvaluationRequest(
                    input=eval_item.input,
                    expected_output=eval_item.expected_output,
                    model_responses=eval_item.model_responses,
                    context=eval_item.context,
                    category=eval_item.category,
                    difficulty=eval_item.difficulty
                )
                
                # Pass selected metrics to evaluation service
                batch_result = await evaluation_service.evaluate_batch(batch_request, request.metrics)
                
                # Store batch result in database
                collection = db.get_collection("batch_evaluations")
                inserted = await collection.insert_one(batch_result.dict(by_alias=True))
                batch_result.id = inserted.inserted_id
                
                # Extract metrics for each model
                responses = []
                
                for model_name, model_result in batch_result.model_results.items():
                    # Get DeepEval scores from metadata
                    deepeval_scores = model_result.metadata.get("deepeval_scores", {})
                    
                    # Get individual metric scores
                    metrics = {}
                    metric_scores = []
                    
                    for metric_name in request.metrics:
                        # Get score from DeepEval results
                        score = deepeval_scores.get(metric_name, 0.0)
                        score = round(score, 2)
                        metrics[metric_name] = score
                        metric_scores.append(score)
                        
                        # Aggregate for metrics_aggregated
                        if model_name not in metric_aggregated_data[metric_name]:
                            metric_aggregated_data[metric_name][model_name] = []
                        metric_aggregated_data[metric_name][model_name].append(score)
                    
                    # Calculate overall score
                    overall_score = round(sum(metric_scores) / len(metric_scores), 2) if metric_scores else 0.0
                    
                    # Store for model averages
                    if model_name not in all_model_scores:
                        all_model_scores[model_name] = []
                    all_model_scores[model_name].append(overall_score)
                    
                    # Create response object
                    response_obj = {
                        "model": model_name,
                        "response": eval_item.model_responses[model_name],
                        "metrics": metrics,
                        "overall_score": overall_score
                    }
                    
                    responses.append(response_obj)
                
                # Create test result
                test_result = {
                    "test_id": f"test-case-{i+1:03d}",
                    "input": eval_item.input,
                    "expected_output": eval_item.expected_output,
                    "responses": responses,
                    "evaluation_id": str(batch_result.id)  # Reference to stored evaluation
                }
                
                test_results.append(test_result)
                
                logger.info(
                    f"Completed evaluation {i+1}/{len(dataset.evaluations)}",
                    test_id=test_result["test_id"],
                    evaluation_id=str(batch_result.id)
                )
            
            # Calculate model averages
            model_average_scores = []
            for model_name, scores in all_model_scores.items():
                avg_score = round(sum(scores) / len(scores), 2)
                model_average_scores.append({
                    "model": model_name,
                    "average_score": avg_score
                })
            
            # Sort by average score descending
            model_average_scores.sort(key=lambda x: x["average_score"], reverse=True)
            
            # Find overall best model
            overall_best_model = model_average_scores[0]["model"] if model_average_scores else None
            
            # Calculate metrics aggregated
            metrics_aggregated = []
            for metric_name in request.metrics:
                metric_scores = {}
                for model_name, scores_list in metric_aggregated_data[metric_name].items():
                    metric_scores[model_name] = round(sum(scores_list) / len(scores_list), 2)
                
                metrics_aggregated.append({
                    "metric": metric_name,
                    "scores": metric_scores
                })
            
            # Create summary
            summary = {
                "total_test_cases": len(test_results),
                "models_evaluated": list(all_model_scores.keys()),
                "overall_best_model": overall_best_model,
                "model_average_scores": model_average_scores,
                "deepeval_metrics_used": request.metrics
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
                message=f"Evaluation completed successfully. {len(test_results)} test cases processed with DeepEval metrics.",
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
    """Create a batch evaluation for multiple models with DeepEval metrics."""
    
    with tracer.start_as_current_span("create_batch_evaluation") as span:
        span.set_attribute("models.count", len(request.model_responses))
        span.set_attribute("selected_metrics", str(metrics))
        
        try:
            # Perform batch evaluation with selected metrics
            result = await evaluation_service.evaluate_batch(request, metrics)
            
            # Store in database
            collection = db.get_collection("batch_evaluations")
            inserted = await collection.insert_one(result.dict(by_alias=True))
            
            # Update result with inserted ID
            result.id = inserted.inserted_id
            
            logger.info(
                "Batch evaluation created",
                evaluation_id=str(result.id),
                models=list(request.model_responses.keys()),
                best_model=result.best_model,
                average_score=result.average_score,
                metrics_used=metrics
            )
            
            return APIResponse(
                message="Batch evaluation created successfully with DeepEval metrics",
                data=result,
                trace_id=getattr(http_request.state, 'trace_id', None)
            )
            
        except Exception as e:
            logger.error("Failed to create batch evaluation", error=str(e), exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create batch evaluation: {str(e)}"
            )


@router.get("/results/{evaluation_id}", response_model=APIResponse)
async def get_evaluation_result(
    evaluation_id: str,
    http_request: Request,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get evaluation results by ID."""
    
    with tracer.start_as_current_span("get_evaluation_result") as span:
        span.set_attribute("evaluation.id", evaluation_id)
        
        try:
            from bson import ObjectId
            if not ObjectId.is_valid(evaluation_id):
                raise HTTPException(status_code=400, detail="Invalid evaluation ID")
            
            collection = db.get_collection("batch_evaluations")
            doc = await collection.find_one({"_id": ObjectId(evaluation_id)})
            
            if not doc:
                raise HTTPException(status_code=404, detail="Evaluation not found")
            
            evaluation = BatchEvaluationResult(**doc)
            
            return APIResponse(
                message="Evaluation result found",
                data=evaluation,
                trace_id=getattr(http_request.state, 'trace_id', None)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get evaluation result", evaluation_id=evaluation_id, error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get evaluation result: {str(e)}"
            )