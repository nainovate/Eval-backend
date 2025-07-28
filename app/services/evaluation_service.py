"""Updated evaluation service aligned with task types and dataset structure."""
import asyncio
import time
from typing import Dict, List, Optional, Any
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


class TaskTypeEvaluationService:
    """Updated evaluation service organized by task types and dataset structure."""
    
    def __init__(self):
        """Initialize all metrics available for any task type."""
        
        # All metrics organized by category - available for ANY task type
        self.metric_categories = {
            'rag-metrics': {
                'answer_relevancy': AnswerRelevancyMetric(threshold=0.7),
                'faithfulness': FaithfulnessMetric(threshold=0.7),
                'contextual_precision': ContextualPrecisionMetric(threshold=0.7),
                'contextual_recall': ContextualRecallMetric(threshold=0.7),
                'contextual_relevancy': ContextualRelevancyMetric(threshold=0.7),
            },
            'safety-ethics': {
                'bias': BiasMetric(threshold=0.5),
                'toxicity': ToxicityMetric(threshold=0.5),
                'hallucination': HallucinationMetric(threshold=0.3),
            },
            'task-specific': {},  # Will add task-specific metrics if available
            'custom-metrics': {}  # Will add G-Eval if available
        }
        
        # Add task-specific metrics if available
        if SUMMARIZATION_AVAILABLE:
            self.metric_categories['task-specific']['summarization'] = SummarizationMetric(threshold=0.7)
        
        # Add custom metrics if available
        if GEVAL_AVAILABLE:
            try:
                self.metric_categories['custom-metrics']['g_eval'] = GEval(
                    name="G-Eval Quality Assessment",
                    criteria="Evaluate the response quality, accuracy, and relevance based on the expected output.",
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
        
        # Flatten all metrics for easy access - ALL metrics available for ANY task type
        self.all_available_metrics = {}
        for category, metrics in self.metric_categories.items():
            self.all_available_metrics.update(metrics)
        
        logger.info(
            "Universal evaluation service initialized - All metrics available for any task type",
            total_metrics=len(self.all_available_metrics),
            metric_categories=list(self.metric_categories.keys()),
            available_metrics=list(self.all_available_metrics.keys()),
            summarization_available=SUMMARIZATION_AVAILABLE,
            geval_available=GEVAL_AVAILABLE
        )
    
    def get_metric_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics organized by category - available for ANY task type."""
        return self.metric_categories
    
    def get_available_metrics(self, category: str = None) -> List[str]:
        """Get list of available metric names, optionally filtered by category."""
        if category and category in self.metric_categories:
            return list(self.metric_categories[category].keys())
        return list(self.all_available_metrics.keys())
    
    def get_metric_info(self) -> Dict[str, Any]:
        """Get detailed information about all available metrics."""
        return {
            "metric_categories": {
                category: {
                    "metrics": list(metrics.keys()),
                    "descriptions": {
                        name: self._get_description(name) 
                        for name in metrics.keys()
                    }
                }
                for category, metrics in self.metric_categories.items()
            },
            "all_available_metrics": list(self.all_available_metrics.keys()),
            "total_count": len(self.all_available_metrics),
            "note": "All metrics can be used with any task type",
            "optional_metrics_status": {
                "summarization": SUMMARIZATION_AVAILABLE,
                "g_eval": GEVAL_AVAILABLE,
            }
        }
    
    def _get_description(self, metric_name: str) -> str:
        """Get description for a metric."""
        descriptions = {
            "answer_relevancy": "How well the answer addresses the input question",
            "faithfulness": "Checks factual grounding based on retrieval context",
            "contextual_precision": "Measures precision of retrieved context",
            "contextual_recall": "Measures recall of retrieved context", 
            "contextual_relevancy": "Assesses relevance of retrieved context to question",
            "bias": "Detects potential bias in responses",
            "toxicity": "Identifies toxic or harmful content",
            "hallucination": "Detects unsupported or fabricated content",
            "g_eval": "Custom evaluation with defined criteria",
            "summarization": "Measures summary quality compared to expected summary",
        }
        return descriptions.get(metric_name, "DeepEval metric")
    
    def _create_test_case_from_dataset_row(self, dataset_row: Dict[str, Any], model_output: str, task_type: str) -> LLMTestCase:
        """Create LLMTestCase from dataset row based on task type - updated for new dataset structure."""
        
        # Task type specific mapping (without generated columns)
        if task_type == 'Question Answering':
            return LLMTestCase(
                input=dataset_row.get('question', ''),
                actual_output=model_output,
                expected_output=dataset_row.get('expected_answer', ''),
                context=dataset_row.get('context', '').split('\n') if dataset_row.get('context') else [],
                retrieval_context=[dataset_row.get('context', '')] if dataset_row.get('context') else []
            )
        
        elif task_type == 'Summarization':
            return LLMTestCase(
                input=dataset_row.get('input_text', ''),
                actual_output=model_output,
                expected_output=dataset_row.get('expected_summary', ''),
                context=[dataset_row.get('input_text', '')],
                retrieval_context=[dataset_row.get('input_text', '')]
            )
        
        elif task_type == 'Conversational QA':
            conversation_history = dataset_row.get('conversation_history', '')
            current_question = dataset_row.get('question', '')
            full_input = f"{conversation_history}\nUser: {current_question}" if conversation_history else current_question
            
            return LLMTestCase(
                input=full_input,
                actual_output=model_output,
                expected_output=dataset_row.get('expected_answer', ''),
                context=dataset_row.get('context', '').split('\n') if dataset_row.get('context') else [conversation_history],
                retrieval_context=[conversation_history, dataset_row.get('context', '')] if dataset_row.get('context') else [conversation_history]
            )
        
        elif task_type == 'Retrieval (RAG)':
            return LLMTestCase(
                input=dataset_row.get('query', ''),
                actual_output=model_output,
                expected_output=dataset_row.get('expected_answer', ''),
                context=dataset_row.get('retrieved_documents', '').split('\n') if dataset_row.get('retrieved_documents') else [],
                retrieval_context=dataset_row.get('retrieved_documents', '').split('\n') if dataset_row.get('retrieved_documents') else []
            )
        
        elif task_type == 'Classification':
            return LLMTestCase(
                input=dataset_row.get('input_text', ''),
                actual_output=model_output,
                expected_output=dataset_row.get('expected_label', ''),
                context=[dataset_row.get('input_text', '')],
                retrieval_context=[dataset_row.get('input_text', '')]
            )
        
        elif task_type == 'Structured Output Generation':
            return LLMTestCase(
                input=dataset_row.get('input_instruction', ''),
                actual_output=model_output,
                expected_output=dataset_row.get('expected_output', ''),
                context=[dataset_row.get('input_instruction', '')],
                retrieval_context=[dataset_row.get('format_schema', '')] if dataset_row.get('format_schema') else []
            )
        
        elif task_type == 'Open-ended Generation':
            return LLMTestCase(
                input=dataset_row.get('prompt', ''),
                actual_output=model_output,
                expected_output=dataset_row.get('reference_output', '') or "No reference provided",
                context=[dataset_row.get('prompt', '')],
                retrieval_context=[dataset_row.get('prompt', '')]
            )
        
        else:
            # Fallback for unknown task types
            # Try to find input and expected output columns dynamically
            input_col = None
            expected_col = None
            
            for key in dataset_row.keys():
                if any(term in key.lower() for term in ['question', 'input', 'prompt', 'query']):
                    input_col = key
                elif any(term in key.lower() for term in ['expected', 'answer', 'summary', 'label']):
                    expected_col = key
            
            return LLMTestCase(
                input=dataset_row.get(input_col, '') if input_col else '',
                actual_output=model_output,
                expected_output=dataset_row.get(expected_col, '') if expected_col else '',
                context=[dataset_row.get(input_col, '')] if input_col else [],
                retrieval_context=[dataset_row.get(input_col, '')] if input_col else []
            )
    
    async def evaluate_dataset_row_multiple_models(
        self, 
        dataset_row: Dict[str, Any],
        model_outputs: Dict[str, str],  # {model_name: generated_output}
        task_type: str,
        selected_metrics: List[str] = None,
        metric_category: str = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate multiple model outputs against the same dataset row."""
        
        increment_active_evaluations()
        with tracer.start_as_current_span("evaluate_multiple_models") as span:
            span.set_attribute("task_type", task_type)
            span.set_attribute("model_count", len(model_outputs))
            span.set_attribute("models", list(model_outputs.keys()))
            
            logger.info(
                "Starting multi-model evaluation",
                task_type=task_type,
                models=list(model_outputs.keys()),
                metrics_requested=selected_metrics or "all_in_category"
            )
            
            try:
                # Determine which metrics to use
                if selected_metrics:
                    metrics_to_run = {
                        name: metric for name, metric in self.all_available_metrics.items() 
                        if name in selected_metrics
                    }
                elif metric_category:
                    metrics_to_run = self.metric_categories.get(metric_category, {})
                else:
                    metrics_to_run = self.all_available_metrics
                
                # Evaluate each model in parallel
                model_tasks = []
                for model_name, model_output in model_outputs.items():
                    task = self._evaluate_single_model_for_row(
                        dataset_row, model_output, model_name, task_type, metrics_to_run
                    )
                    model_tasks.append((model_name, task))
                
                # Run all model evaluations in parallel
                results = await asyncio.gather(
                    *[task for _, task in model_tasks], 
                    return_exceptions=True
                )
                
                # Process results
                model_scores = {}
                for i, (model_name, _) in enumerate(model_tasks):
                    if isinstance(results[i], Exception):
                        logger.error(f"Model '{model_name}' evaluation failed", error=str(results[i]))
                        model_scores[model_name] = {}
                    else:
                        model_scores[model_name] = results[i]
                
                # Calculate comparative metrics
                comparison_stats = self._calculate_model_comparison_stats(model_scores)
                
                logger.info(
                    "Multi-model evaluation completed",
                    task_type=task_type,
                    models_evaluated=len(model_scores),
                    best_model=comparison_stats.get('best_overall_model'),
                    metrics_count=len(metrics_to_run)
                )
                
                return {
                    'model_scores': model_scores,
                    'comparison_stats': comparison_stats
                }
                
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    "Multi-model evaluation failed",
                    task_type=task_type,
                    models=list(model_outputs.keys()),
                    error=str(e),
                    exc_info=True
                )
                raise
            finally:
                decrement_active_evaluations()
    
    async def _evaluate_single_model_for_row(
        self, 
        dataset_row: Dict[str, Any],
        model_output: str,
        model_name: str,
        task_type: str,
        metrics_to_run: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate a single model's output for a dataset row."""
        
        with tracer.start_as_current_span(f"evaluate_model_{model_name}") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("task_type", task_type)
            
            try:
                # Create test case from dataset row
                test_case = self._create_test_case_from_dataset_row(dataset_row, model_output, task_type)
                
                # Run evaluation
                scores = await self._run_metrics_on_test_case(test_case, metrics_to_run)
                
                span.set_attribute("model.scores_count", len(scores))
                span.set_attribute("model.avg_score", sum(scores.values()) / len(scores) if scores else 0)
                
                return scores
                
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Single model evaluation failed for {model_name}", error=str(e))
                raise
    
    def _calculate_model_comparison_stats(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate comparative statistics across models."""
        
        if not model_scores:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for scores in model_scores.values():
            all_metrics.update(scores.keys())
        
        # Calculate stats per metric
        metric_stats = {}
        for metric in all_metrics:
            metric_values = []
            model_metric_scores = {}
            
            for model_name, scores in model_scores.items():
                if metric in scores:
                    score = scores[metric]
                    metric_values.append(score)
                    model_metric_scores[model_name] = score
            
            if metric_values:
                best_model = max(model_metric_scores.items(), key=lambda x: x[1])
                worst_model = min(model_metric_scores.items(), key=lambda x: x[1])
                
                metric_stats[metric] = {
                    'best_model': best_model[0],
                    'best_score': best_model[1],
                    'worst_model': worst_model[0], 
                    'worst_score': worst_model[1],
                    'average_score': sum(metric_values) / len(metric_values),
                    'score_range': max(metric_values) - min(metric_values),
                    'all_scores': model_metric_scores
                }
        
        # Calculate overall best model (average across all metrics)
        overall_averages = {}
        for model_name, scores in model_scores.items():
            if scores:
                overall_averages[model_name] = sum(scores.values()) / len(scores)
        
        best_overall = max(overall_averages.items(), key=lambda x: x[1]) if overall_averages else None
        
        return {
            'metric_stats': metric_stats,
            'best_overall_model': best_overall[0] if best_overall else None,
            'best_overall_score': best_overall[1] if best_overall else None,
            'model_overall_averages': overall_averages,
            'total_models_evaluated': len(model_scores),
            'total_metrics_evaluated': len(all_metrics)
        }
    
    async def evaluate_dataset_batch_multiple_models(
        self,
        dataset_rows: List[Dict[str, Any]],
        model_outputs: Dict[str, List[str]],  # {model_name: [output1, output2, ...]}
        task_type: str,
        selected_metrics: List[str] = None,
        metric_category: str = None
    ) -> Dict[str, Any]:
        """Evaluate multiple models across multiple dataset rows."""
        
        increment_active_evaluations()
        with tracer.start_as_current_span("evaluate_batch_multiple_models") as span:
            span.set_attribute("task_type", task_type)
            span.set_attribute("batch_size", len(dataset_rows))
            span.set_attribute("model_count", len(model_outputs))
            
            logger.info(
                "Starting batch multi-model evaluation",
                task_type=task_type,
                batch_size=len(dataset_rows),
                models=list(model_outputs.keys())
            )
            
            try:
                # Validate input lengths match
                model_names = list(model_outputs.keys())
                for model_name, outputs in model_outputs.items():
                    if len(outputs) != len(dataset_rows):
                        raise ValueError(f"Model '{model_name}' output count ({len(outputs)}) doesn't match dataset size ({len(dataset_rows)})")
                
                # Process each row
                row_tasks = []
                for i, dataset_row in enumerate(dataset_rows):
                    # Get outputs for this row from all models
                    row_model_outputs = {
                        model_name: outputs[i] 
                        for model_name, outputs in model_outputs.items()
                    }
                    
                    task = self.evaluate_dataset_row_multiple_models(
                        dataset_row=dataset_row,
                        model_outputs=row_model_outputs,
                        task_type=task_type,
                        selected_metrics=selected_metrics,
                        metric_category=metric_category
                    )
                    row_tasks.append(task)
                
                # Run all rows in parallel
                row_results = await asyncio.gather(*row_tasks, return_exceptions=True)
                
                # Aggregate results
                successful_results = []
                failed_rows = []
                
                for i, result in enumerate(row_results):
                    if isinstance(result, Exception):
                        logger.error(f"Row {i} evaluation failed", error=str(result))
                        failed_rows.append(i)
                    else:
                        successful_results.append(result)
                
                # Calculate batch-level statistics
                batch_stats = self._calculate_batch_level_stats(successful_results, model_names)
                
                logger.info(
                    "Batch multi-model evaluation completed",
                    successful_rows=len(successful_results),
                    failed_rows=len(failed_rows),
                    best_model_overall=batch_stats.get('best_model_overall')
                )
                
                return {
                    'row_results': successful_results,
                    'batch_statistics': batch_stats,
                    'successful_rows': len(successful_results),
                    'failed_rows': failed_rows,
                    'models_evaluated': model_names
                }
                
            except Exception as e:
                span.record_exception(e)
                logger.error(
                    "Batch multi-model evaluation failed",
                    error=str(e),
                    exc_info=True
                )
                raise
            finally:
                decrement_active_evaluations()
    
    def _calculate_batch_level_stats(self, row_results: List[Dict], model_names: List[str]) -> Dict[str, Any]:
        """Calculate statistics across the entire batch for all models."""
        
        if not row_results:
            return {}
        
        # Aggregate scores for each model across all rows
        model_aggregated_scores = {model_name: {} for model_name in model_names}
        
        for row_result in row_results:
            model_scores = row_result.get('model_scores', {})
            for model_name, scores in model_scores.items():
                if model_name not in model_aggregated_scores:
                    model_aggregated_scores[model_name] = {}
                
                for metric, score in scores.items():
                    if metric not in model_aggregated_scores[model_name]:
                        model_aggregated_scores[model_name][metric] = []
                    model_aggregated_scores[model_name][metric].append(score)
        
        # Calculate averages and statistics for each model
        model_final_stats = {}
        for model_name, metric_lists in model_aggregated_scores.items():
            model_stats = {}
            total_avg_scores = []
            
            for metric, score_list in metric_lists.items():
                if score_list:
                    avg_score = sum(score_list) / len(score_list)
                    model_stats[metric] = {
                        'average': avg_score,
                        'min': min(score_list),
                        'max': max(score_list),
                        'count': len(score_list)
                    }
                    total_avg_scores.append(avg_score)
            
            # Overall average for this model
            if total_avg_scores:
                model_final_stats[model_name] = {
                    'metric_stats': model_stats,
                    'overall_average': sum(total_avg_scores) / len(total_avg_scores),
                    'metrics_evaluated': len(model_stats)
                }
        
        # Find best performing model overall
        best_model = None
        best_score = -1
        for model_name, stats in model_final_stats.items():
            if stats['overall_average'] > best_score:
                best_score = stats['overall_average']
                best_model = model_name
        
        return {
            'model_statistics': model_final_stats,
            'best_model_overall': best_model,
            'best_score_overall': best_score,
            'total_rows_processed': len(row_results),
            'models_compared': len(model_final_stats)
        }
    
    async def _run_metrics_on_test_case(self, test_case: LLMTestCase, metrics_dict: Dict[str, Any]) -> Dict[str, float]:
        """Run metrics on a test case."""
        scores = {}
        
        if not metrics_dict:
            logger.warning("No metrics provided for evaluation")
            return scores
        
        logger.info(f"Running {len(metrics_dict)} metrics", metrics=list(metrics_dict.keys()))
        
        # Run all metrics in parallel
        tasks = []
        for metric_name, metric in metrics_dict.items():
            tasks.append(self._evaluate_single_metric(metric_name, metric, test_case))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            metric_name = list(metrics_dict.keys())[i]
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
                    f"Metric '{metric_name}' completed",
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
    
    # Legacy methods for backward compatibility
    async def evaluate_response(self, request: EvaluationRequest, selected_metrics: List[str] = None) -> EvaluationResult:
        """Legacy method for backward compatibility."""
        # Convert to dataset row format
        dataset_row = {
            'question': request.prompt,
            'expected_answer': request.expected_output or ''
        }
        
        scores = await self.evaluate_dataset_row(
            dataset_row=dataset_row,
            model_output=request.response,
            task_type='Question Answering',  # Default task type
            selected_metrics=selected_metrics
        )
        
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        return EvaluationResult(
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
            evaluation_duration_ms=0,
            metadata={**request.metadata, "deepeval_scores": scores}
        )


# Create global service instance
evaluation_service = TaskTypeEvaluationService()