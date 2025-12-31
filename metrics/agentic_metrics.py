"""
Agentic-Specific Metrics for the Text2Cypher pipeline.

Provides metrics specific to the agentic loop:
- Iteration tracking
- Recovery rates
- Error analysis
- Improvement per iteration

These metrics help evaluate the effectiveness of the self-correction mechanism.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from core.agent_state import AgentState, ErrorType


@dataclass
class AgenticMetrics:
    """Metrics specific to the agentic loop."""

    # Basic counts
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0

    # Iteration metrics
    avg_iterations: float = 0.0
    max_iterations_used: int = 0
    first_attempt_success_rate: float = 0.0

    # Recovery metrics
    recovery_rate: float = 0.0  # % of initially failed that succeeded after refinement
    iterations_to_success: Dict[int, int] = field(default_factory=dict)

    # Error analysis
    error_type_distribution: Dict[str, int] = field(default_factory=dict)
    error_recovery_by_type: Dict[str, float] = field(default_factory=dict)

    # Improvement tracking
    improvement_per_iteration: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_questions": self.total_questions,
            "successful_questions": self.successful_questions,
            "failed_questions": self.failed_questions,
            "success_rate": self.successful_questions / self.total_questions if self.total_questions > 0 else 0,
            "avg_iterations": self.avg_iterations,
            "max_iterations_used": self.max_iterations_used,
            "first_attempt_success_rate": self.first_attempt_success_rate,
            "recovery_rate": self.recovery_rate,
            "iterations_to_success": self.iterations_to_success,
            "error_type_distribution": self.error_type_distribution,
            "error_recovery_by_type": self.error_recovery_by_type,
            "improvement_per_iteration": self.improvement_per_iteration,
        }


def calculate_agentic_metrics(states: List[AgentState]) -> AgenticMetrics:
    """
    Calculate agentic-specific metrics from a list of final states.

    Args:
        states: List of completed AgentState objects

    Returns:
        AgenticMetrics object with computed metrics
    """
    if not states:
        return AgenticMetrics()

    total = len(states)
    successful = sum(1 for s in states if s.success)
    failed = total - successful

    # Iteration statistics
    iterations = [s.total_iterations for s in states]
    avg_iterations = float(np.mean(iterations)) if iterations else 0.0
    max_iterations = max(iterations) if iterations else 0

    # First attempt success
    first_attempt_success = sum(
        1 for s in states
        if s.attempts and s.attempts[0].is_valid
    )
    first_attempt_rate = first_attempt_success / total if total > 0 else 0.0

    # Recovery rate (succeeded after initial failure)
    initially_failed = [s for s in states if s.attempts and not s.attempts[0].is_valid]
    recovered = sum(1 for s in initially_failed if s.success)
    recovery_rate = recovered / len(initially_failed) if initially_failed else 0.0

    # Iterations to success distribution
    iter_to_success: Dict[int, int] = defaultdict(int)
    for s in states:
        if s.success:
            for i, attempt in enumerate(s.attempts):
                if attempt.is_valid:
                    iter_to_success[i + 1] += 1
                    break

    # Error type analysis
    error_distribution: Dict[str, int] = defaultdict(int)
    error_counts: Dict[str, int] = defaultdict(int)
    error_recovery: Dict[str, int] = defaultdict(int)

    for s in states:
        for attempt in s.attempts:
            if not attempt.is_valid and attempt.primary_error:
                error_type = attempt.primary_error.value
                error_distribution[error_type] += 1

        # Track recovery by error type (first error)
        if s.attempts and not s.attempts[0].is_valid:
            first_error = s.attempts[0].primary_error
            if first_error:
                error_type = first_error.value
                error_counts[error_type] += 1
                if s.success:
                    error_recovery[error_type] += 1

    # Calculate recovery rates by error type
    error_recovery_rates = {
        e: error_recovery[e] / error_counts[e] if error_counts[e] > 0 else 0.0
        for e in error_counts
    }

    # Improvement per iteration (cumulative success rate)
    max_iter = max(s.max_iterations for s in states) if states else 0
    improvement = []
    for i in range(max_iter):
        success_by_iter = sum(
            1 for s in states
            if any(attempt.is_valid for attempt in s.attempts[:i + 1])
        )
        improvement.append(success_by_iter / total if total > 0 else 0.0)

    return AgenticMetrics(
        total_questions=total,
        successful_questions=successful,
        failed_questions=failed,
        avg_iterations=avg_iterations,
        max_iterations_used=max_iterations,
        first_attempt_success_rate=first_attempt_rate,
        recovery_rate=recovery_rate,
        iterations_to_success=dict(iter_to_success),
        error_type_distribution=dict(error_distribution),
        error_recovery_by_type=error_recovery_rates,
        improvement_per_iteration=improvement,
    )


def calculate_llmetric_q(
    pass_at_1: bool,
    kg_valid: bool,
    jaccard_output: float,
    jaro_winkler: float,
    rouge_l_f1: float,
) -> float:
    """
    Calculate LLMetric-Q (question-level composite metric).

    Formula:
    LLMetric-Q = w1 * Pass@1 + w2 * KG_Valid + w3 * Jaccard_Output + w4 * JaRou

    Where:
    - w1 = 0.3 (Pass@1 weight)
    - w2 = 0.4 (KG validity weight)
    - w3 = 0.2 (Jaccard output weight)
    - w4 = 0.1 (JaRou score weight)

    Args:
        pass_at_1: Whether output matches exactly
        kg_valid: Whether query is structurally valid
        jaccard_output: Output similarity score
        jaro_winkler: Jaro-Winkler similarity
        rouge_l_f1: Rouge-L F1 score

    Returns:
        LLMetric-Q score (0-100)
    """
    w1 = 0.3  # Pass@1
    w2 = 0.4  # KG Validity
    w3 = 0.2  # Jaccard Output
    w4 = 0.1  # JaRou

    pass_1_score = 100.0 if pass_at_1 else 0.0
    kg_valid_score = 100.0 if kg_valid else 0.0
    jaccard_output_score = jaccard_output * 100.0
    jarou_score = ((jaro_winkler + rouge_l_f1) / 2) * 100.0

    # If perfect match, return 100
    if pass_1_score == 100.0 and kg_valid_score == 100.0:
        return 100.0

    llmetric_q = (
        w1 * pass_1_score +
        w2 * kg_valid_score +
        w3 * jaccard_output_score +
        w4 * jarou_score
    )

    return llmetric_q


def calculate_llmetric(
    pass_at_1_rate: float,
    kg_valid_rate: float,
    jaccard_output_avg: float,
    jarou_avg: float,
) -> float:
    """
    Calculate LLMetric (model-level composite metric).

    Formula:
    LLMetric = w1 * Pass@1_Rate + w2 * KG_Valid_Rate + w3 * Jaccard_Avg + w4 * JaRou_Avg

    Args:
        pass_at_1_rate: Pass@1 rate (0-100)
        kg_valid_rate: KG validity rate (0-100)
        jaccard_output_avg: Average Jaccard output score (0-100)
        jarou_avg: Average JaRou score (0-100)

    Returns:
        LLMetric score (0-100)
    """
    w1 = 0.3  # Pass@1
    w2 = 0.4  # KG Validity
    w3 = 0.2  # Jaccard Output
    w4 = 0.1  # JaRou

    return (
        w1 * pass_at_1_rate +
        w2 * kg_valid_rate +
        w3 * jaccard_output_avg +
        w4 * jarou_avg
    )


def compare_with_baseline(
    agentic_metrics: AgenticMetrics,
    baseline_pass_at_1: float,
    baseline_kg_valid: float,
    baseline_llmetric: float,
) -> Dict[str, Any]:
    """
    Compare agentic results with baseline (non-agentic) results.

    Args:
        agentic_metrics: Metrics from agentic pipeline
        baseline_pass_at_1: Baseline Pass@1 rate
        baseline_kg_valid: Baseline KG validity rate
        baseline_llmetric: Baseline LLMetric score

    Returns:
        Comparison dictionary
    """
    agentic_pass_at_1 = (
        agentic_metrics.successful_questions / agentic_metrics.total_questions * 100
        if agentic_metrics.total_questions > 0 else 0
    )

    return {
        "baseline_pass_at_1": baseline_pass_at_1,
        "agentic_pass_at_1": agentic_pass_at_1,
        "pass_at_1_improvement": agentic_pass_at_1 - baseline_pass_at_1,
        "baseline_kg_valid": baseline_kg_valid,
        "baseline_llmetric": baseline_llmetric,
        "recovery_rate": agentic_metrics.recovery_rate,
        "avg_iterations": agentic_metrics.avg_iterations,
        "first_attempt_matches_baseline": (
            agentic_metrics.first_attempt_success_rate * 100 - baseline_pass_at_1
        ),
    }
