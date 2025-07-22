"""Evaluation service using only DeepEval built-in metrics."""
import asyncio
import time
from typing import Dict, List, Optional
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric
)
from app.telemetry.metrics import (
    record_evaluation_metrics, 
    increment_active_evaluations, 
    decrement_active_evaluations
)

from deepeval.test_case import LLMTestCase
import structlog

from app.models.evaluation import EvaluationRequest, EvaluationResult, BatchEvaluationRequest, BatchEvaluationResult
from app.telemetry.setup import get_tracer

logger = structlog.get_logger()
tracer = get_tracer(__name__)


class EvaluationService:
    """Service for performing AI model evaluations using only DeepEval built-in metrics."""
    
    def __init__(self):
        # Only real DeepEval built-in metrics
        self.available_metrics = {
            "answer_relevancy": AnswerRelevancyMetric(threshold=0.7),
            "faithfulness": FaithfulnessMetric(threshold=0.7),
            "contextual_precision": ContextualPrecisionMetric(threshold=0.7),
            "contextual_recall": ContextualRecallMetric(threshold=0.7),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=0.7),
            "bias": BiasMetric(threshold=0.5),
            "toxicity": ToxicityMetric(threshold=0.5),
            "hallucination": HallucinationMetric(threshold=0.3)
        }
    
    async def evaluate_response(
        self, 
        request: EvaluationRequest,
        selected_metrics: List[str] = None
    ) -> EvaluationResult:
        """Evaluate a single model response with selected DeepEval metrics."""
        increment_active_evaluations()  
        with tracer.start_as_current_span("evaluate_response") as span:
            start_time = time.time()
            
            span.set_attribute("model.name", request.model_name)
            span.set_attribute("evaluation.type", request.evaluation_type)
            span.set_attribute("selected_metrics", str(selected_metrics))
            
            logger.info(
                "Starting evaluation",
                model_name=request.model_name,
                evaluation_type=request.evaluation_type,
                selected_metrics=selected_metrics
            )
            
            try:
                # Create test case with all required fields
                test_case = LLMTestCase(
                    input=request.prompt,
                    actual_output=request.response,
                    expected_output=request.expected_output,
                    context=[request.prompt, request.expected_output] if request.expected_output else [request.prompt],
                    retrieval_context=[request.prompt]
                )
                
                # Run evaluation with selected metrics
                scores = await self._run_evaluation(test_case, selected_metrics or [])
                
                # Calculate overall score
                overall_score = sum(scores.values()) / len(scores) if scores else 0.0
                
                # Create result
                result = EvaluationResult(
                    model_name=request.model_name,
                    prompt=request.prompt,
                    response=request.response,
                    expected_output=request.expected_output,
                    accuracy_score=scores.get("contextual_precision", 0.0),
                    relevancy_score=scores.get("answer_relevancy", 0.0),
                    coherence_score=scores.get("contextual_relevancy", 0.0),
                    fluency_score=scores.get("faithfulness", 0.0),
                    overall_score=overall_score,
                    evaluation_type=request.evaluation_type,
                    evaluation_duration_ms=(time.time() - start_time) * 1000,
                    metadata={**request.metadata, "deepeval_scores": scores}
                )
                
                span.set_attribute("evaluation.overall_score", overall_score)
                 # After getting scores, record metrics
                for metric_name, score in scores.items():
                    record_evaluation_metrics(
                        model_name=request.model_name,
                        metric_name=metric_name,
                        score=score,
                        duration=(time.time() - start_time)
                    )
                
                logger.info(
                    "Evaluation completed",
                    model_name=request.model_name,
                    overall_score=overall_score,
                    duration_ms=result.evaluation_duration_ms
                )
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    "Evaluation failed",
                    model_name=request.model_name,
                    error=str(e),
                    exc_info=True
                )
                raise
            finally:
                decrement_active_evaluations()  # Add this
    
    async def _run_evaluation(self, test_case: LLMTestCase, selected_metrics: List[str]) -> Dict[str, float]:
        """Run selected DeepEval metrics on test case."""
        scores = {}
        
        # Filter to only available metrics
        valid_metrics = {}
        for metric_name in selected_metrics:
            if metric_name in self.available_metrics:
                valid_metrics[metric_name] = self.available_metrics[metric_name]
            else:
                logger.warning(f"Unknown DeepEval metric: {metric_name}")
        
        if not valid_metrics:
            logger.warning("No valid DeepEval metrics selected")
            return {}
        
        logger.info(f"Running {len(valid_metrics)} DeepEval metrics", metrics=list(valid_metrics.keys()))
        
        # Run metrics in parallel
        tasks = []
        for metric_name, metric in valid_metrics.items():
            tasks.append(self._evaluate_metric(metric_name, metric, test_case))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            metric_name = list(valid_metrics.keys())[i]
            if isinstance(result, Exception):
                logger.warning(f"DeepEval metric {metric_name} failed", error=str(result))
                scores[metric_name] = 0.0
            else:
                scores[metric_name] = result
        
        return scores
    
    async def _evaluate_metric(
        self, 
        metric_name: str, 
        metric, 
        test_case: LLMTestCase
    ) -> float:
        """Evaluate a single DeepEval metric."""
        try:
            # Run metric evaluation in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, metric.measure, test_case)
            score = getattr(metric, 'score', 0.0)
            logger.info(f"DeepEval metric {metric_name} completed", score=score)
            return score
        except Exception as e:
            logger.warning(f"DeepEval metric {metric_name} evaluation failed", error=str(e))
            return 0.0
    
    async def evaluate_batch(
        self, 
        request: BatchEvaluationRequest,
        selected_metrics: List[str] = None
    ) -> BatchEvaluationResult:
        """Evaluate multiple model responses with selected DeepEval metrics."""
        
        with tracer.start_as_current_span("evaluate_batch") as span:
            start_time = time.time()
            
            span.set_attribute("input.length", len(request.input))
            span.set_attribute("models.count", len(request.model_responses))
            span.set_attribute("selected_metrics", str(selected_metrics))
            
            logger.info(
                "Starting batch evaluation",
                input_preview=request.input[:100] + "..." if len(request.input) > 100 else request.input,
                models=list(request.model_responses.keys()),
                selected_metrics=selected_metrics
            )
            
            try:
                # Evaluate each model response
                model_results = {}
                evaluation_tasks = []
                
                for model_name, response in request.model_responses.items():
                    eval_request = EvaluationRequest(
                        model_name=model_name,
                        prompt=request.input,
                        response=response,
                        expected_output=request.expected_output,
                        evaluation_type=request.category or "general",
                        metadata={
                            "context": request.context,
                            "difficulty": request.difficulty,
                            **request.metadata
                        }
                    )
                    evaluation_tasks.append(
                        self._evaluate_single_model(model_name, eval_request, selected_metrics)
                    )
                
                # Run evaluations in parallel
                results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
                
                # Process results
                valid_scores = []
                for i, result in enumerate(results):
                    model_name = list(request.model_responses.keys())[i]
                    if isinstance(result, Exception):
                        logger.warning(f"Model {model_name} evaluation failed", error=str(result))
                        # Create a default result for failed evaluations
                        model_results[model_name] = EvaluationResult(
                            model_name=model_name,
                            prompt=request.input,
                            response=request.model_responses[model_name],
                            expected_output=request.expected_output,
                            overall_score=0.0,
                            evaluation_type=request.category or "general"
                        )
                    else:
                        model_results[model_name] = result
                        if result.overall_score is not None:
                            valid_scores.append(result.overall_score)
                
                # Calculate comparative metrics
                best_model = None
                worst_model = None
                average_score = None
                score_variance = None
                
                if valid_scores:
                    average_score = sum(valid_scores) / len(valid_scores)
                    
                    # Find best and worst performing models
                    best_score = max(valid_scores)
                    worst_score = min(valid_scores)
                    
                    for model_name, result in model_results.items():
                        if result.overall_score == best_score:
                            best_model = model_name
                        if result.overall_score == worst_score:
                            worst_model = model_name
                    
                    # Calculate variance
                    if len(valid_scores) > 1:
                        score_variance = sum((score - average_score) ** 2 for score in valid_scores) / len(valid_scores)
                
                # Create batch result
                batch_result = BatchEvaluationResult(
                    dataset_name="batch_evaluation",
                    input=request.input,
                    expected_output=request.expected_output,
                    context=request.context,
                    category=request.category,
                    difficulty=request.difficulty,
                    model_results=model_results,
                    best_model=best_model,
                    worst_model=worst_model,
                    average_score=average_score,
                    score_variance=score_variance,
                    evaluation_duration_ms=(time.time() - start_time) * 1000
                )
                
                span.set_attribute("evaluation.best_model", best_model or "none")
                span.set_attribute("evaluation.average_score", average_score or 0.0)
                
                logger.info(
                    "Batch evaluation completed",
                    models=list(request.model_responses.keys()),
                    best_model=best_model,
                    average_score=average_score,
                    duration_ms=batch_result.evaluation_duration_ms
                )
                
                return batch_result
                
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    "Batch evaluation failed",
                    models=list(request.model_responses.keys()),
                    error=str(e),
                    exc_info=True
                )
                raise
    
    async def _evaluate_single_model(
        self, 
        model_name: str, 
        request: EvaluationRequest,
        selected_metrics: List[str] = None
    ) -> EvaluationResult:
        """Helper method to evaluate a single model with selected DeepEval metrics."""
        return await self.evaluate_response(request, selected_metrics)


# Global service instance
evaluation_service = EvaluationService()