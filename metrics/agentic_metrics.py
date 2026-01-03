"""
Agentic-Specific Metrics for the Text2Cypher pipeline.

This module provides metrics based on established research literature:

## Metric Sources:
- Pass@1, Pass@k: HumanEval Benchmark (OpenAI, Kulal et al. 2019)
- EX (Execution Accuracy): Spider/BIRD Benchmarks
- KG Valid: kg-axel baseline research
- Refinement Gain, Recovery Rate: Self-Refine (Madaan et al., 2023)

## Metrics Provided:
1. Pass@1 Rate: First attempt success percentage
2. Pass@k Rate: Final success percentage after k iterations
3. KG Valid@1 Rate: First attempt KG validity (comparable to kg-axel)
4. KG Valid@k Rate: Final KG validity after refinement
5. Refinement Gain: Pass@k - Pass@1 (absolute improvement)
6. Recovery Rate: (Pass@k - Pass@1) / (1 - Pass@1) (relative improvement)
7. Iteration statistics and error analysis
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from core.agent_state import AgentState, ErrorType


@dataclass
class AgenticMetrics:
    """
    Comprehensive metrics for the agentic self-correction loop.

    Uses formal naming from established research literature.
    """

    # === BASIC COUNTS ===
    total_questions: int = 0
    successful_questions: int = 0
    failed_questions: int = 0

    # === PASS@1 METRICS (First Attempt - Baseline Comparable) ===
    pass_at_1_count: int = 0
    pass_at_1_rate: float = 0.0  # HumanEval: First attempt success rate
    kg_valid_at_1_count: int = 0
    kg_valid_at_1_rate: float = 0.0  # kg-axel comparable: First attempt KG validity

    # === PASS@k METRICS (Final - After Refinement) ===
    pass_at_k_count: int = 0
    pass_at_k_rate: float = 0.0  # HumanEval: Success within k iterations
    kg_valid_at_k_count: int = 0
    kg_valid_at_k_rate: float = 0.0  # Final KG validity after refinement

    # === SELF-REFINE METRICS (Improvement from Refinement) ===
    # Source: Self-Refine (Madaan et al., 2023)
    refinement_gain: float = 0.0  # Pass@k - Pass@1 (absolute improvement)
    recovery_rate: float = 0.0  # (Pass@k - Pass@1) / (1 - Pass@1) (relative)
    kg_refinement_gain: float = 0.0  # KG_Valid@k - KG_Valid@1
    kg_recovery_rate: float = 0.0  # KG validity recovery rate

    # === ITERATION METRICS ===
    avg_iterations: float = 0.0
    max_iterations_used: int = 0
    iterations_to_success: Dict[int, int] = field(default_factory=dict)

    # === ERROR ANALYSIS ===
    error_type_distribution: Dict[str, int] = field(default_factory=dict)
    error_recovery_by_type: Dict[str, float] = field(default_factory=dict)

    # === IMPROVEMENT CURVE ===
    improvement_per_iteration: List[float] = field(default_factory=list)

    # === BACKWARD COMPATIBILITY ===
    first_attempt_success_rate: float = 0.0  # Alias for pass_at_1_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            # Basic counts
            "total_questions": self.total_questions,
            "successful_questions": self.successful_questions,
            "failed_questions": self.failed_questions,

            # Pass@1 metrics (baseline comparable)
            "pass_at_1_count": self.pass_at_1_count,
            "pass_at_1_rate": self.pass_at_1_rate,
            "kg_valid_at_1_count": self.kg_valid_at_1_count,
            "kg_valid_at_1_rate": self.kg_valid_at_1_rate,

            # Pass@k metrics (after refinement)
            "pass_at_k_count": self.pass_at_k_count,
            "pass_at_k_rate": self.pass_at_k_rate,
            "kg_valid_at_k_count": self.kg_valid_at_k_count,
            "kg_valid_at_k_rate": self.kg_valid_at_k_rate,

            # Self-Refine metrics
            "refinement_gain": self.refinement_gain,
            "recovery_rate": self.recovery_rate,
            "kg_refinement_gain": self.kg_refinement_gain,
            "kg_recovery_rate": self.kg_recovery_rate,

            # Iteration metrics
            "avg_iterations": self.avg_iterations,
            "max_iterations_used": self.max_iterations_used,
            "iterations_to_success": self.iterations_to_success,

            # Error analysis
            "error_type_distribution": self.error_type_distribution,
            "error_recovery_by_type": self.error_recovery_by_type,

            # Improvement curve
            "improvement_per_iteration": self.improvement_per_iteration,

            # Backward compatibility
            "success_rate": self.pass_at_k_rate,
            "first_attempt_success_rate": self.pass_at_1_rate,
        }


def calculate_agentic_metrics(states: List[AgentState]) -> AgenticMetrics:
    """
    Calculate comprehensive agentic metrics from a list of final states.

    Uses formal metric naming from established research:
    - Pass@1, Pass@k: HumanEval Benchmark (OpenAI, 2021)
    - KG Valid: kg-axel baseline research
    - Refinement Gain, Recovery Rate: Self-Refine (Madaan et al., 2023)

    Args:
        states: List of completed AgentState objects

    Returns:
        AgenticMetrics object with all computed metrics
    """
    if not states:
        return AgenticMetrics()

    total = len(states)

    # === BASIC COUNTS ===
    successful = sum(1 for s in states if s.success)
    failed = total - successful

    # === PASS@1 METRICS (First Attempt - Baseline Comparable) ===
    pass_at_1_count = sum(1 for s in states if s.pass_at_1)
    pass_at_1_rate = (pass_at_1_count / total * 100) if total > 0 else 0.0

    kg_valid_at_1_count = sum(1 for s in states if s.kg_valid_at_1)
    kg_valid_at_1_rate = (kg_valid_at_1_count / total * 100) if total > 0 else 0.0

    # === PASS@k METRICS (Final - After Refinement) ===
    pass_at_k_count = successful  # Same as success count
    pass_at_k_rate = (pass_at_k_count / total * 100) if total > 0 else 0.0

    kg_valid_at_k_count = sum(1 for s in states if s.kg_valid_at_k)
    kg_valid_at_k_rate = (kg_valid_at_k_count / total * 100) if total > 0 else 0.0

    # === SELF-REFINE METRICS (Improvement from Refinement) ===
    # Refinement Gain = Pass@k - Pass@1 (absolute improvement in percentage points)
    refinement_gain = pass_at_k_rate - pass_at_1_rate

    # Recovery Rate = (Pass@k - Pass@1) / (1 - Pass@1)
    # Measures: Of the queries that failed at first, what % were recovered?
    initially_failed_count = total - pass_at_1_count
    recovered_count = sum(1 for s in states if s.was_recovered)
    recovery_rate = (recovered_count / initially_failed_count * 100) if initially_failed_count > 0 else 0.0

    # KG Refinement Gain = KG_Valid@k - KG_Valid@1
    kg_refinement_gain = kg_valid_at_k_rate - kg_valid_at_1_rate

    # KG Recovery Rate
    kg_initially_invalid_count = total - kg_valid_at_1_count
    kg_recovered_count = sum(1 for s in states if s.was_kg_recovered)
    kg_recovery_rate = (kg_recovered_count / kg_initially_invalid_count * 100) if kg_initially_invalid_count > 0 else 0.0

    # === ITERATION METRICS ===
    iterations = [s.total_iterations for s in states]
    avg_iterations = float(np.mean(iterations)) if iterations else 0.0
    max_iterations = max(iterations) if iterations else 0

    # Iterations to success distribution
    iter_to_success: Dict[int, int] = defaultdict(int)
    for s in states:
        if s.success:
            for i, attempt in enumerate(s.attempts):
                if attempt.is_valid:
                    iter_to_success[i + 1] += 1
                    break

    # === ERROR ANALYSIS ===
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
        e: (error_recovery[e] / error_counts[e] * 100) if error_counts[e] > 0 else 0.0
        for e in error_counts
    }

    # === IMPROVEMENT CURVE (Cumulative success rate per iteration) ===
    max_iter = max(s.max_iterations for s in states) if states else 0
    improvement = []
    for i in range(max_iter):
        success_by_iter = sum(
            1 for s in states
            if any(attempt.is_valid for attempt in s.attempts[:i + 1])
        )
        improvement.append((success_by_iter / total * 100) if total > 0 else 0.0)

    return AgenticMetrics(
        # Basic counts
        total_questions=total,
        successful_questions=successful,
        failed_questions=failed,

        # Pass@1 metrics
        pass_at_1_count=pass_at_1_count,
        pass_at_1_rate=pass_at_1_rate,
        kg_valid_at_1_count=kg_valid_at_1_count,
        kg_valid_at_1_rate=kg_valid_at_1_rate,

        # Pass@k metrics
        pass_at_k_count=pass_at_k_count,
        pass_at_k_rate=pass_at_k_rate,
        kg_valid_at_k_count=kg_valid_at_k_count,
        kg_valid_at_k_rate=kg_valid_at_k_rate,

        # Self-Refine metrics
        refinement_gain=refinement_gain,
        recovery_rate=recovery_rate,
        kg_refinement_gain=kg_refinement_gain,
        kg_recovery_rate=kg_recovery_rate,

        # Iteration metrics
        avg_iterations=avg_iterations,
        max_iterations_used=max_iterations,
        iterations_to_success=dict(iter_to_success),

        # Error analysis
        error_type_distribution=dict(error_distribution),
        error_recovery_by_type=error_recovery_rates,

        # Improvement curve
        improvement_per_iteration=improvement,

        # Backward compatibility
        first_attempt_success_rate=pass_at_1_rate,
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

    This function provides a comprehensive comparison using formal metric names
    from established research literature.

    Args:
        agentic_metrics: Metrics from agentic pipeline
        baseline_pass_at_1: Baseline Pass@1 rate (from kg-axel)
        baseline_kg_valid: Baseline KG validity rate (from kg-axel)
        baseline_llmetric: Baseline LLMetric score (from kg-axel)

    Returns:
        Comparison dictionary with improvement metrics
    """
    return {
        # === BASELINE METRICS (from kg-axel) ===
        "baseline_pass_at_1": baseline_pass_at_1,
        "baseline_kg_valid": baseline_kg_valid,
        "baseline_llmetric": baseline_llmetric,

        # === AGENTIC PASS@1 METRICS (Comparable to Baseline) ===
        "agentic_pass_at_1": agentic_metrics.pass_at_1_rate,
        "agentic_kg_valid_at_1": agentic_metrics.kg_valid_at_1_rate,

        # === AGENTIC PASS@k METRICS (After Refinement) ===
        "agentic_pass_at_k": agentic_metrics.pass_at_k_rate,
        "agentic_kg_valid_at_k": agentic_metrics.kg_valid_at_k_rate,

        # === SELF-REFINE METRICS (Improvement from Refinement) ===
        "refinement_gain": agentic_metrics.refinement_gain,
        "recovery_rate": agentic_metrics.recovery_rate,
        "kg_refinement_gain": agentic_metrics.kg_refinement_gain,
        "kg_recovery_rate": agentic_metrics.kg_recovery_rate,

        # === BASELINE COMPARISON ===
        # How does first attempt compare to baseline?
        "pass_at_1_vs_baseline": agentic_metrics.pass_at_1_rate - baseline_pass_at_1,
        "kg_valid_at_1_vs_baseline": agentic_metrics.kg_valid_at_1_rate - baseline_kg_valid,

        # How does final result (after refinement) compare to baseline?
        "pass_at_k_vs_baseline": agentic_metrics.pass_at_k_rate - baseline_pass_at_1,
        "kg_valid_at_k_vs_baseline": agentic_metrics.kg_valid_at_k_rate - baseline_kg_valid,

        # === ITERATION STATS ===
        "avg_iterations": agentic_metrics.avg_iterations,
        "max_iterations_used": agentic_metrics.max_iterations_used,
    }


def create_metrics_summary_table(states: List[AgentState]) -> Dict[str, Any]:
    """
    Create a summary table with all formal metrics for reporting.

    Returns a dictionary suitable for DataFrame creation or display.

    Args:
        states: List of completed AgentState objects

    Returns:
        Dictionary with metric names and values
    """
    metrics = calculate_agentic_metrics(states)

    return {
        "Metric": [
            # Pass@1 (First Attempt)
            "Pass@1 Rate (%)",
            "KG Valid@1 Rate (%)",
            # Pass@k (After Refinement)
            "Pass@k Rate (%)",
            "KG Valid@k Rate (%)",
            # Self-Refine Improvement
            "Refinement Gain (pp)",
            "Recovery Rate (%)",
            "KG Refinement Gain (pp)",
            "KG Recovery Rate (%)",
            # Iteration Stats
            "Avg Iterations",
            "Max Iterations Used",
        ],
        "Value": [
            round(metrics.pass_at_1_rate, 2),
            round(metrics.kg_valid_at_1_rate, 2),
            round(metrics.pass_at_k_rate, 2),
            round(metrics.kg_valid_at_k_rate, 2),
            round(metrics.refinement_gain, 2),
            round(metrics.recovery_rate, 2),
            round(metrics.kg_refinement_gain, 2),
            round(metrics.kg_recovery_rate, 2),
            round(metrics.avg_iterations, 2),
            metrics.max_iterations_used,
        ],
        "Source": [
            "HumanEval (OpenAI, 2021)",
            "kg-axel (Baseline)",
            "HumanEval (OpenAI, 2021)",
            "kg-axel (Baseline)",
            "Self-Refine (Madaan, 2023)",
            "Self-Refine (Madaan, 2023)",
            "Self-Refine (Madaan, 2023)",
            "Self-Refine (Madaan, 2023)",
            "-",
            "-",
        ],
    }
