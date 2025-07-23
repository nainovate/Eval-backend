"""Working evaluation service - fixed variable scope issues."""
import asyncio
import time
from typing import Dict, List, Optional
import structlog

# Import core DeepEval metrics
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    BiasMetric,
    ToxicityMetric,
    HallucinationMetric,
)

# Try to import additional metrics with proper scope
SUMMARIZATION_AVAILABLE = False
GEVAL_AVAILABLE = False
RAGAS_AVAILABLE = False

try:
    from deepeval.metrics import SummarizationMetric
    SUMMARIZATION_AVAILABLE = True
    print("✅ SummarizationMetric available")
except ImportError:
    print("⚠️ SummarizationMetric not available")

try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    GEVAL_AVAILABLE = True
    print("✅ GEval available")
except ImportError:
    try:
        from deepeval.metrics import GEvalMetric as GEval
        from deepeval.test_case import LLMTestCaseParams
        GEVAL_AVAILABLE = True
        print("✅ GEvalMetric available")
    except ImportError:
        print("⚠️ G-Eval not available")

try:
    from deepeval.metrics import RagasMetric
    RAGAS_AVAILABLE = True
    print("✅ RagasMetric available")
except ImportError:
    print("⚠️ RagasMetric not available")

from deepeval.test_case import LLMTestCase
from app.models.evaluation import EvaluationRequest, EvaluationResult, BatchEvaluationRequest, BatchEvaluationResult
from app.telemetry.setup import get_tracer
from app.telemetry.metrics import (
    record_evaluation_metrics, 
    increment_active_evaluations, 
    decrement_active_evaluations
)

logger = structlog.get_logger()
tracer = get_tracer(__name__)


class EvaluationService:
    """Fixed evaluation service with all required methods."""
    
    def __init__(self):
        """Initialize metrics with proper scope management."""
        
        # Core metrics that definitely exist
        self.available_metrics = {
            "answer_relevancy": AnswerRelevancyMetric(threshold=0.7),
            "faithfulness": FaithfulnessMetric(threshold=0.7),
            "contextual_precision": ContextualPrecisionMetric(threshold=0.7),
            "contextual_recall": ContextualRecallMetric(threshold=0.7),
            "contextual_relevancy": ContextualRelevancyMetric(threshold=0.7),
            "bias": BiasMetric(threshold=0.5),
            "toxicity": ToxicityMetric(threshold=0.5),
            "hallucination": HallucinationMetric(threshold=0.3),
        }
        
        # Add optional metrics if available
        if SUMMARIZATION_AVAILABLE:
            self.available_metrics["summarization"] = SummarizationMetric(threshold=0.7)
        
        # Fixed G-Eval configuration with proper error handling
        if GEVAL_AVAILABLE:
            try:
                self.available_metrics["g_eval"] = GEval(
                    name="G-Eval Quality Assessment",
                    criteria="Evaluate the response quality, accuracy, and relevance.",
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.EXPECTED_OUTPUT
                    ],
                    threshold=0.7
                )
                print("✅ G-Eval configured successfully")
            except Exception as e:
                print(f"⚠️ Failed to configure G-Eval: {e}")
                # Remove g_eval from available metrics if configuration fails
                pass
        
        if RAGAS_AVAILABLE:
            self.available_metrics["ragas"] = RagasMetric(threshold=0.7)
        
        logger.info(
            "Working DeepEval service initialized",
            total_metrics=len(self.available_metrics),
            available_metrics=list(self.available_metrics.keys()),
            summarization_available=SUMMARIZATION_AVAILABLE,
            geval_available=GEVAL_AVAILABLE,
            ragas_available=RAGAS_AVAILABLE
        )
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available metric names."""
        return list(self.available_metrics.keys())
    
    def get_metric_info(self) -> Dict[str, any]:
        """Get detailed information about available metrics."""
        return {
            "available_metrics": list(self.available_metrics.keys()),
            "total_count": len(self.available_metrics),
            "optional_metrics_status": {
                "summarization": SUMMARIZATION_AVAILABLE,
                "g_eval": GEVAL_AVAILABLE,
                "ragas": RAGAS_AVAILABLE
            },
            "metric_details": {
                name: {
                    "threshold": getattr(metric, 'threshold', 0.7),
                    "description": self._get_description(name)
                }
                for name, metric in self.available_metrics.items()
            }
        }
    
    def _get_description(self, metric_name: str) -> str:
        """Get description for a metric."""
        descriptions = {
            "answer_relevancy": "How well the answer addresses the input prompt",
            "faithfulness": "Checks factual grounding based on retrieval context",
            "contextual_precision": "Measures relevance of retrieved context",
            "contextual_recall": "Measures completeness of retrieved context", 
            "contextual_relevancy": "Assesses usefulness of retrieved context",
            "bias": "Detects potential bias in responses",
            "toxicity": "Identifies toxic or harmful content",
            "hallucination": "Detects unsupported or fabricated content",
            "g_eval": "Custom evaluation with defined criteria",
            "summarization": "Measures summary quality compared to ground truth",
            "ragas": "Composite RAG evaluation score"
        }
        return descriptions.get(metric_name, "DeepEval metric")
    
    async def evaluate_response(
        self, 
        request: EvaluationRequest,
        selected_metrics: List[str] = None
    ) -> EvaluationResult:
        """Evaluate a single model response with specified metrics."""
        
        increment_active_evaluations()
        with tracer.start_as_current_span("evaluate_response_working") as span:
            start_time = time.time()
            
            span.set_attribute("model.name", request.model_name)
            span.set_attribute("evaluation.type", request.evaluation_type)
            span.set_attribute("selected_metrics", str(selected_metrics))
            
            logger.info(
                "Starting working DeepEval evaluation",
                model_name=request.model_name,
                selected_metrics=selected_metrics
            )
            
            try:
                # Create test case with proper expected_output handling and retrieval_context
                retrieval_context = []
                if request.expected_output:
                    retrieval_context.append(request.expected_output)
                retrieval_context.append(request.prompt)
                
                test_case = LLMTestCase(
                    input=request.prompt,
                    actual_output=request.response,
                    expected_output=request.expected_output or "No expected output provided",
                    context=[request.prompt, request.expected_output] if request.expected_output else [request.prompt],
                    retrieval_context=retrieval_context
                )
                
                # Run evaluation with specified metrics
                scores = await self._run_metrics(test_case, selected_metrics)
                
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
                    metadata={
                        **request.metadata, 
                        "deepeval_scores": scores,
                        "metrics_used": selected_metrics
                    }
                )
                
                span.set_attribute("evaluation.overall_score", overall_score)
                span.set_attribute("evaluation.metrics_count", len(scores))
                
                # Record individual metrics
                for metric_name, score in scores.items():
                    record_evaluation_metrics(
                        model_name=request.model_name,
                        metric_name=metric_name,
                        score=score,
                        duration=(time.time() - start_time)
                    )
                
                logger.info(
                    "Working DeepEval evaluation completed",
                    model_name=request.model_name,
                    overall_score=overall_score,
                    metrics_evaluated=len(scores),
                    duration_ms=result.evaluation_duration_ms
                )
                
                return result
                
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    "Working DeepEval evaluation failed",
                    model_name=request.model_name,
                    error=str(e),
                    exc_info=True
                )
                raise
            finally:
                decrement_active_evaluations()
    
    async def _run_metrics(self, test_case: LLMTestCase, selected_metrics: List[str]) -> Dict[str, float]:
        """Run the specified metrics directly."""
        scores = {}
        
        # Filter to only available metrics (case-insensitive)
        valid_metrics = {}
        for metric_name in selected_metrics:
            # Handle case variations
            metric_key = metric_name.lower()
            if metric_key in self.available_metrics:
                valid_metrics[metric_key] = self.available_metrics[metric_key]
            else:
                logger.warning(f"Metric '{metric_name}' not available. Available: {list(self.available_metrics.keys())}")
        
        if not valid_metrics:
            logger.warning("No valid metrics found, using defaults")
            # Use only core metrics that are guaranteed to work
            default_metrics = ["answer_relevancy", "bias", "toxicity"]
            for metric in default_metrics:
                if metric in self.available_metrics:
                    valid_metrics[metric] = self.available_metrics[metric]
        
        logger.info(f"Running {len(valid_metrics)} DeepEval metrics", metrics=list(valid_metrics.keys()))
        
        # Run all metrics in parallel
        tasks = []
        for metric_name, metric in valid_metrics.items():
            tasks.append(self._evaluate_single_metric(metric_name, metric, test_case))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            metric_name = list(valid_metrics.keys())[i]
            if isinstance(result, Exception):
                logger.warning(f"Metric '{metric_name}' failed", error=str(result))
                scores[metric_name] = 0.0
            else:
                scores[metric_name] = result
        
        return scores
    
    async def _evaluate_single_metric(self, metric_name: str, metric, test_case: LLMTestCase) -> float:
        """Evaluate a single metric with better error handling."""
        
        with tracer.start_as_current_span(f"metric_{metric_name}") as span:
            span.set_attribute("metric.name", metric_name)
            
            try:
                start_time = time.time()
                
                # Run the DeepEval metric
                await metric.a_measure(test_case)
                score = metric.score
                
                duration_ms = (time.time() - start_time) * 1000
                threshold = getattr(metric, 'threshold', 0.7)
                
                span.set_attribute("metric.score", score)
                span.set_attribute("metric.duration_ms", duration_ms)
                span.set_attribute("metric.threshold", threshold)
                span.set_attribute("metric.passed", score >= threshold)
                
                logger.info(
                    f"DeepEval metric '{metric_name}' completed",
                    score=score,
                    threshold=threshold,
                    passed=score >= threshold,
                    duration_ms=round(duration_ms, 2)
                )
                
                return score
                
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Metric '{metric_name}' evaluation failed", error=str(e), exc_info=True)
                raise
    
    async def evaluate_batch(
        self, 
        request: BatchEvaluationRequest, 
        selected_metrics: List[str] = None
    ) -> BatchEvaluationResult:
        """Evaluate multiple models with specified metrics."""
        
        with tracer.start_as_current_span("evaluate_batch_working") as span:
            start_time = time.time()
            
            model_names = list(request.model_responses.keys())
            span.set_attribute("batch.model_count", len(model_names))
            span.set_attribute("batch.metrics", str(selected_metrics))
            
            logger.info(
                "Starting working batch evaluation",
                models=model_names,
                selected_metrics=selected_metrics
            )
            
            model_results = {}
            
            # Evaluate each model
            for model_name, response in request.model_responses.items():
                eval_request = EvaluationRequest(
                    model_name=model_name,
                    prompt=request.input,
                    response=response,
                    expected_output=request.expected_output,
                    evaluation_type=request.category,
                    metadata={"batch_id": str(time.time())}
                )
                
                result = await self.evaluate_response(eval_request, selected_metrics)
                model_results[model_name] = result
            
            # Find best model
            best_model = max(
                model_results.items(), 
                key=lambda x: x[1].overall_score
            )[0] if model_results else None
            
            duration = time.time() - start_time
            
            logger.info(
                "Working batch evaluation completed",
                models=model_names,
                best_model=best_model,
                duration_ms=duration * 1000
            )
            
            # Fixed: Include all required fields for BatchEvaluationResult
            return BatchEvaluationResult(
                dataset_name="batch_evaluation",  # Required field
                input=request.input,
                expected_output=request.expected_output,
                context=request.context,
                category=request.category,
                difficulty=request.difficulty,
                model_results=model_results,
                best_model=best_model,
                evaluation_duration_ms=duration * 1000
            )


# Create global service instance
evaluation_service = EvaluationService()