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
    success: bool = False

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

    @property
    def first_attempt_success(self) -> bool:
        """Check if the first attempt was successful."""
        if self.attempts:
            return self.attempts[0].is_valid
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
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
            "success": self.success,
            "first_attempt_success": self.first_attempt_success,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    def to_row(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for DataFrame row."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth_query,
            "reasoning_level": self.question_metadata.get("Tingkat Penalaran", ""),
            "sublevel": self.question_metadata.get("Sublevel", ""),
            "complexity": self.question_metadata.get("Tingkat Kompleksitas", ""),
            "prompt_type": self.prompt_type,
            "schema_type": self.schema_type,
            "final_query": self.final_query,
            "success": self.success,
            "total_iterations": self.total_iterations,
            "first_attempt_success": self.first_attempt_success,
            "errors_encountered": [e.value for e in self.get_all_errors()],
        }
