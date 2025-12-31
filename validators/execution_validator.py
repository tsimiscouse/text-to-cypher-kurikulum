"""
Execution Validator for Cypher queries.

Validates that a Cypher query can be executed without errors
and optionally checks for empty results.
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import Driver

from core.agent_state import ValidationResult, ErrorType

logger = logging.getLogger(__name__)


class ExecutionValidator:
    """
    Validates Cypher query execution.

    Runs the query against Neo4j and checks for:
    1. Execution errors (syntax, runtime, etc.)
    2. Empty results (potentially incorrect query)
    """

    def __init__(self, driver: Driver):
        """
        Initialize the execution validator.

        Args:
            driver: Neo4j database driver
        """
        self.driver = driver

    def validate(
        self,
        query: str,
        ground_truth_query: Optional[str] = None,
        check_empty: bool = True,
    ) -> ValidationResult:
        """
        Validate query execution.

        Args:
            query: Cypher query to execute
            ground_truth_query: Optional ground truth for comparison
            check_empty: Whether to flag empty results as potential error

        Returns:
            ValidationResult with execution status
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = [dict(record) for record in result]

            # Check for empty results
            if check_empty and len(records) == 0:
                # If ground truth is provided, check if it returns results
                if ground_truth_query:
                    gt_has_results = self._check_ground_truth_has_results(
                        ground_truth_query
                    )
                    if gt_has_results:
                        return ValidationResult(
                            validator_name="execution",
                            is_valid=False,
                            error_type=ErrorType.EMPTY_RESULT,
                            error_message="Query returned empty result but expected non-empty",
                            metadata={
                                "result_count": 0,
                                "expected_results": True,
                            },
                        )

            return ValidationResult(
                validator_name="execution",
                is_valid=True,
                error_type=None,
                error_message=None,
                metadata={
                    "result_count": len(records),
                    "results": records[:5],  # Store first 5 results for debugging
                },
            )

        except Exception as e:
            error_msg = str(e)
            logger.debug(f"Execution error: {error_msg}")

            return ValidationResult(
                validator_name="execution",
                is_valid=False,
                error_type=ErrorType.EXECUTION_ERROR,
                error_message=self._format_error_message(error_msg),
                metadata={"raw_error": error_msg},
            )

    def _check_ground_truth_has_results(self, query: str) -> bool:
        """Check if the ground truth query returns any results."""
        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = list(result)
                return len(records) > 0
        except Exception:
            return False

    def _format_error_message(self, error: str) -> str:
        """Format the error message for better readability."""
        # Common Neo4j error patterns
        if "SyntaxError" in error:
            return f"Cypher syntax error: {error}"
        elif "TypeError" in error:
            return f"Type error in query: {error}"
        elif "ParameterMissing" in error:
            return f"Missing parameter: {error}"
        elif "ConstraintValidation" in error:
            return f"Constraint violation: {error}"
        else:
            return f"Execution error: {error}"


class ResultValidator:
    """
    Validates query results against ground truth.

    Compares the output of the generated query with the expected output.
    """

    def __init__(self, driver: Driver):
        """
        Initialize the result validator.

        Args:
            driver: Neo4j database driver
        """
        self.driver = driver

    def validate(
        self,
        generated_query: str,
        ground_truth_query: str,
    ) -> ValidationResult:
        """
        Validate that generated query produces same results as ground truth.

        Args:
            generated_query: The generated Cypher query
            ground_truth_query: The expected correct query

        Returns:
            ValidationResult with comparison status
        """
        try:
            # Execute both queries
            gen_results = self._execute_query(generated_query)
            gt_results = self._execute_query(ground_truth_query)

            # Compare results
            is_match = self._compare_results(gen_results, gt_results)

            if is_match:
                return ValidationResult(
                    validator_name="result_match",
                    is_valid=True,
                    error_type=None,
                    error_message=None,
                    metadata={
                        "generated_count": len(gen_results),
                        "ground_truth_count": len(gt_results),
                        "exact_match": True,
                    },
                )
            else:
                return ValidationResult(
                    validator_name="result_match",
                    is_valid=False,
                    error_type=ErrorType.EMPTY_RESULT,
                    error_message="Query results do not match ground truth",
                    metadata={
                        "generated_count": len(gen_results),
                        "ground_truth_count": len(gt_results),
                        "exact_match": False,
                    },
                )

        except Exception as e:
            return ValidationResult(
                validator_name="result_match",
                is_valid=False,
                error_type=ErrorType.EXECUTION_ERROR,
                error_message=str(e),
                metadata={},
            )

    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    def _compare_results(
        self, gen_results: List[Dict], gt_results: List[Dict]
    ) -> bool:
        """
        Compare two result sets.

        Results are considered equal if they contain the same data,
        regardless of order.
        """
        if len(gen_results) != len(gt_results):
            return False

        # Convert to comparable format (handle Neo4j Node/Relationship objects)
        gen_normalized = self._normalize_results(gen_results)
        gt_normalized = self._normalize_results(gt_results)

        # Sort for comparison
        gen_sorted = sorted(gen_normalized, key=lambda x: str(x))
        gt_sorted = sorted(gt_normalized, key=lambda x: str(x))

        return gen_sorted == gt_sorted

    def _normalize_results(self, results: List[Dict]) -> List[Dict]:
        """Normalize results for comparison."""
        normalized = []
        for record in results:
            norm_record = {}
            for key, value in record.items():
                # Handle Neo4j Node objects
                if hasattr(value, "labels") and hasattr(value, "items"):
                    norm_record[key] = dict(value)
                # Handle Neo4j Relationship objects
                elif hasattr(value, "type") and hasattr(value, "items"):
                    norm_record[key] = dict(value)
                else:
                    norm_record[key] = value
            normalized.append(norm_record)
        return normalized
