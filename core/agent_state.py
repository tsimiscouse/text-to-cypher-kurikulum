"""
Agent State Management for the Agentic Text2Cypher pipeline.

This module defines the data structures for tracking the state of the
agentic loop during query generation and refinement.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class AgentStatus(Enum):
    """Status of the agentic loop."""
    GENERATING = "generating"
    VALIDATING = "validating"
    REFINING = "refining"
    SUCCESS = "success"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    FAILED = "failed"


class ErrorType(Enum):
    """Types of errors encountered during validation."""
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    PROPERTIES_ERROR = "properties_error"
    EXECUTION_ERROR = "execution_error"
    EMPTY_RESULT = "empty_result"
    NO_ERROR = "no_error"


@dataclass
class ValidationResult:
    """Result from a single validator."""
    validator_name: str
    is_valid: bool
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validator_name": self.validator_name,
            "is_valid": self.is_valid,
            "error_type": self.error_type.value if self.error_type else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class Attempt:
    """Records a single attempt in the agentic loop."""
    iteration: int
    generated_query: str
    reasoning: Optional[str] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    feedback_given: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        if not self.validation_results:
            return False
        return all(v.is_valid for v in self.validation_results)

    @property
    def primary_error(self) -> Optional[ErrorType]:
        """Get the first error encountered (validation order matters)."""
        for v in self.validation_results:
            if not v.is_valid and v.error_type:
                return v.error_type
        return None

    @property
    def error_message(self) -> Optional[str]:
        """Get the first error message."""
        for v in self.validation_results:
            if not v.is_valid and v.error_message:
                return v.error_message
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "generated_query": self.generated_query,
            "reasoning": self.reasoning,
            "is_valid": self.is_valid,
            "primary_error": self.primary_error.value if self.primary_error else None,
            "error_message": self.error_message,
            "validation_results": [v.to_dict() for v in self.validation_results],
            "feedback_given": self.feedback_given,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentState:
    """
    Maintains the complete state of the agentic loop for one question.

    This is the central data structure passed through all stages of
    the agentic loop.

    ## Metric Naming Convention (Based on Established Research):

    ### From HumanEval/Codex (OpenAI, Kulal et al. 2019):
    - Pass@1: First attempt success rate
    - Pass@k: Success within k attempts

    ### From Spider/BIRD Benchmarks:
    - EX (Execution Accuracy): Final query correctness

    ### From kg-axel (Baseline Research):
    - KG Valid: Query passes syntax + schema + properties validation

    ### From Self-Refine (Madaan et al., 2023):
    - Refinement Gain: Pass@k - Pass@1
    - Recovery Rate: (Pass@k - Pass@1) / (1 - Pass@1)
    """
    # Input
    question_id: int
    question: str
    ground_truth_query: str
    question_metadata: Dict[str, str] = field(default_factory=dict)

    # Configuration
    prompt_type: str = ""
    schema_type: str = ""
    max_iterations: int = 3

    # State
    current_iteration: int = 0
    current_query: Optional[str] = None
    current_reasoning: Optional[str] = None
    status: AgentStatus = AgentStatus.GENERATING

    # History
    attempts: List[Attempt] = field(default_factory=list)

    # Results (populated at end)
    final_query: Optional[str] = None
    final_validation: Optional[List[ValidationResult]] = None
    total_iterations: int = 0

    # === PRIMARY METRICS (Formal Naming) ===
    # Pass@k / EX (Execution Accuracy): Final success after k iterations
    success: bool = False

    # KG Valid@k: Final query passes KG validation (syntax + schema + properties)
    kg_valid: bool = False

    def add_attempt(self, attempt: Attempt) -> None:
        """Add an attempt to history."""
        self.attempts.append(attempt)
        self.total_iterations = len(self.attempts)

    def get_previous_attempt(self) -> Optional[Attempt]:
        """Get the most recent attempt."""
        return self.attempts[-1] if self.attempts else None

    def get_all_errors(self) -> List[ErrorType]:
        """Get all unique errors across all attempts."""
        errors = set()
        for attempt in self.attempts:
            if attempt.primary_error:
                errors.add(attempt.primary_error)
        return list(errors)

    # =========================================================================
    # PASS@1 METRICS (First Attempt - Comparable to Baseline)
    # =========================================================================

    @property
    def pass_at_1(self) -> bool:
        """
        Pass@1: First attempt output matches ground truth exactly.

        Source: HumanEval Benchmark (OpenAI, Kulal et al. 2019)
        Definition: Probability that the first generated solution is correct.

        This is the primary metric for comparing with non-agentic baselines.
        """
        if self.attempts:
            return self.attempts[0].is_valid
        return False

    @property
    def first_attempt_success(self) -> bool:
        """Alias for pass_at_1 for backward compatibility."""
        return self.pass_at_1

    @property
    def kg_valid_at_1(self) -> bool:
        """
        KG Valid@1: First attempt passes KG validation.

        Source: kg-axel baseline research
        Definition: Query passes syntax + schema + properties validation.
        Note: Execution errors do NOT affect KG Valid (matches kg-axel definition).

        This is the comparable metric for baseline (kg-axel) comparison.
        kg-axel's "Fully Correct" = syntax_correct & schema_correct & properties_correct
        """
        if not self.attempts:
            return False

        first_attempt = self.attempts[0]
        kg_errors = [ErrorType.SYNTAX_ERROR, ErrorType.SCHEMA_ERROR, ErrorType.PROPERTIES_ERROR]

        # Check if first attempt has any syntax/schema/properties errors
        has_kg_error = any(
            v.error_type in kg_errors
            for v in first_attempt.validation_results
            if not v.is_valid
        )
        return not has_kg_error

    @property
    def first_attempt_kg_valid(self) -> bool:
        """Alias for kg_valid_at_1 for backward compatibility."""
        return self.kg_valid_at_1

    # =========================================================================
    # PASS@k METRICS (Final Attempt - After Refinement)
    # =========================================================================

    @property
    def pass_at_k(self) -> bool:
        """
        Pass@k: Success within k iterations (where k = max_iterations).

        Source: HumanEval Benchmark (OpenAI, Kulal et al. 2019)
        Definition: Probability that at least one of k attempts is correct.

        Alias for `success` field.
        """
        return self.success

    @property
    def execution_accuracy(self) -> bool:
        """
        EX (Execution Accuracy): Final query produces correct output.

        Source: Spider/BIRD Benchmarks
        Definition: Whether execution result of predicted query matches ground truth.

        Alias for `success` field.
        """
        return self.success

    @property
    def kg_valid_at_k(self) -> bool:
        """
        KG Valid@k: Final query passes KG validation after refinement.

        Alias for `kg_valid` field.
        """
        return self.kg_valid

    # =========================================================================
    # SELF-REFINE METRICS (Improvement from Refinement)
    # =========================================================================

    @property
    def refinement_gain(self) -> float:
        """
        Refinement Gain (Self-Refine Δ): Improvement from self-correction.

        Source: Self-Refine (Madaan et al., 2023)
        Formula: Pass@k - Pass@1 (as float: 1.0 or 0.0)

        Returns:
            1.0 if query was corrected (failed at 1, succeeded at k)
            0.0 if no change or degradation
            -1.0 if somehow degraded (shouldn't happen in practice)
        """
        pass_1 = 1.0 if self.pass_at_1 else 0.0
        pass_k = 1.0 if self.pass_at_k else 0.0
        return pass_k - pass_1

    @property
    def kg_valid_refinement_gain(self) -> float:
        """
        KG Valid Refinement Gain: Improvement in KG validity from refinement.

        Formula: KG_Valid@k - KG_Valid@1
        """
        kg_1 = 1.0 if self.kg_valid_at_1 else 0.0
        kg_k = 1.0 if self.kg_valid_at_k else 0.0
        return kg_k - kg_1

    @property
    def was_recovered(self) -> bool:
        """
        Check if this query was recovered through refinement.

        True if: failed at Pass@1 but succeeded at Pass@k
        Used for calculating Recovery Rate at aggregate level.
        """
        return not self.pass_at_1 and self.pass_at_k

    @property
    def was_kg_recovered(self) -> bool:
        """
        Check if KG validity was recovered through refinement.

        True if: invalid at KG_Valid@1 but valid at KG_Valid@k
        """
        return not self.kg_valid_at_1 and self.kg_valid_at_k

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            # Metadata
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth_query": self.ground_truth_query,
            "question_metadata": self.question_metadata,
            "prompt_type": self.prompt_type,
            "schema_type": self.schema_type,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "status": self.status.value,
            "final_query": self.final_query,
            "total_iterations": self.total_iterations,

            # === PASS@1 METRICS (First Attempt - Baseline Comparable) ===
            "pass_at_1": self.pass_at_1,
            "kg_valid_at_1": self.kg_valid_at_1,

            # === PASS@k METRICS (Final - After Refinement) ===
            "pass_at_k": self.pass_at_k,
            "execution_accuracy": self.execution_accuracy,
            "kg_valid_at_k": self.kg_valid_at_k,

            # === SELF-REFINE METRICS (Improvement) ===
            "refinement_gain": self.refinement_gain,
            "kg_valid_refinement_gain": self.kg_valid_refinement_gain,
            "was_recovered": self.was_recovered,
            "was_kg_recovered": self.was_kg_recovered,

            # === BACKWARD COMPATIBILITY ===
            "success": self.success,
            "kg_valid": self.kg_valid,
            "first_attempt_success": self.first_attempt_success,
            "first_attempt_kg_valid": self.first_attempt_kg_valid,

            # History
            "attempts": [a.to_dict() for a in self.attempts],
        }

    def to_row(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for DataFrame row."""
        return {
            # Metadata
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth_query,
            "reasoning_level": self.question_metadata.get("Tingkat Penalaran", ""),
            "sublevel": self.question_metadata.get("Sublevel", ""),
            "complexity": self.question_metadata.get("Tingkat Kompleksitas", ""),
            "prompt_type": self.prompt_type,
            "schema_type": self.schema_type,
            "final_query": self.final_query,
            "total_iterations": self.total_iterations,

            # === PASS@1 METRICS (First Attempt - Baseline Comparable) ===
            "pass_at_1": self.pass_at_1,
            "kg_valid_at_1": self.kg_valid_at_1,

            # === PASS@k METRICS (Final - After Refinement) ===
            "pass_at_k": self.pass_at_k,
            "execution_accuracy": self.execution_accuracy,
            "kg_valid_at_k": self.kg_valid_at_k,

            # === SELF-REFINE METRICS (Improvement) ===
            "refinement_gain": self.refinement_gain,
            "kg_valid_refinement_gain": self.kg_valid_refinement_gain,
            "was_recovered": self.was_recovered,
            "was_kg_recovered": self.was_kg_recovered,

            # === BACKWARD COMPATIBILITY ===
            "success": self.success,
            "kg_valid": self.kg_valid,
            "first_attempt_success": self.first_attempt_success,
            "first_attempt_kg_valid": self.first_attempt_kg_valid,

            # Error tracking
            "errors_encountered": [e.value for e in self.get_all_errors()],
        }
