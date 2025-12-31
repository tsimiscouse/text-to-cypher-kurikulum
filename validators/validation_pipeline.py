"""
Validation Pipeline for Cypher queries.

Orchestrates all validators in sequence:
1. Syntax → 2. Schema → 3. Properties → 4. Execution → 5. Result Check
"""
import logging
from typing import List, Tuple, Optional
from neo4j import Driver

from core.agent_state import ValidationResult, ErrorType
from .syntax_validator import SyntaxValidatorWrapper
from .schema_validator import SchemaValidatorWrapper
from .properties_validator import PropertiesValidatorWrapper
from .execution_validator import ExecutionValidator, ResultValidator

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Orchestrates all validators in sequence.

    Validation order (short-circuit on first failure for efficiency):
    1. Syntax → 2. Schema → 3. Properties → 4. Execution → 5. Result Check

    This order ensures that we catch the most fundamental errors first
    and provide the most actionable feedback.
    """

    def __init__(
        self,
        driver: Driver,
        stop_on_first_error: bool = True,
        check_empty_results: bool = True,
    ):
        """
        Initialize the validation pipeline.

        Args:
            driver: Neo4j database driver
            stop_on_first_error: If True, stop validation on first error
            check_empty_results: If True, flag empty results as potential error
        """
        self.driver = driver
        self.stop_on_first_error = stop_on_first_error
        self.check_empty_results = check_empty_results

        # Initialize validators
        self.syntax_validator = SyntaxValidatorWrapper(driver)
        self.schema_validator = SchemaValidatorWrapper(driver)
        self.properties_validator = PropertiesValidatorWrapper(driver)
        self.execution_validator = ExecutionValidator(driver)
        self.result_validator = ResultValidator(driver)

    def validate(
        self,
        generated_query: str,
        ground_truth_query: Optional[str] = None,
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validators on the generated query.

        Args:
            generated_query: The Cypher query to validate
            ground_truth_query: Optional ground truth for result comparison

        Returns:
            Tuple of (all_valid, list of ValidationResult)
        """
        results: List[ValidationResult] = []

        # Check for empty query
        if not generated_query or not generated_query.strip():
            results.append(
                ValidationResult(
                    validator_name="syntax",
                    is_valid=False,
                    error_type=ErrorType.SYNTAX_ERROR,
                    error_message="Empty query",
                    metadata={},
                )
            )
            return False, results

        # 1. Syntax Validation
        logger.debug("Running syntax validation...")
        syntax_result = self.syntax_validator.validate(generated_query)
        results.append(syntax_result)

        if self.stop_on_first_error and not syntax_result.is_valid:
            logger.debug(f"Syntax validation failed: {syntax_result.error_message}")
            return False, results

        # 2. Schema Validation (only if syntax passed)
        if syntax_result.is_valid:
            logger.debug("Running schema validation...")
            schema_result = self.schema_validator.validate(generated_query)
            results.append(schema_result)

            if self.stop_on_first_error and not schema_result.is_valid:
                logger.debug(f"Schema validation failed: {schema_result.error_message}")
                return False, results

        # 3. Properties Validation (only if syntax passed)
        if syntax_result.is_valid:
            logger.debug("Running properties validation...")
            props_result = self.properties_validator.validate(generated_query)
            results.append(props_result)

            if self.stop_on_first_error and not props_result.is_valid:
                logger.debug(
                    f"Properties validation failed: {props_result.error_message}"
                )
                return False, results

        # 4. Execution Validation (only if all structural validations passed)
        structural_valid = all(r.is_valid for r in results)
        if structural_valid:
            logger.debug("Running execution validation...")
            exec_result = self.execution_validator.validate(
                generated_query,
                ground_truth_query=ground_truth_query,
                check_empty=self.check_empty_results,
            )
            results.append(exec_result)

            if self.stop_on_first_error and not exec_result.is_valid:
                logger.debug(f"Execution validation failed: {exec_result.error_message}")
                return False, results

        # All validations passed
        all_valid = all(r.is_valid for r in results)
        logger.debug(f"Validation complete: all_valid={all_valid}")

        return all_valid, results

    def validate_full(
        self,
        generated_query: str,
        ground_truth_query: str,
    ) -> Tuple[bool, List[ValidationResult], dict]:
        """
        Run full validation including result comparison.

        Args:
            generated_query: The Cypher query to validate
            ground_truth_query: The expected correct query

        Returns:
            Tuple of (all_valid, list of ValidationResult, metrics_dict)
        """
        # Run standard validation
        all_valid, results = self.validate(generated_query, ground_truth_query)

        # Calculate additional metrics
        metrics = {
            "syntax_valid": False,
            "schema_valid": False,
            "properties_valid": False,
            "execution_valid": False,
            "kg_valid": False,
            "pass_at_1": False,
        }

        for result in results:
            if result.validator_name == "syntax":
                metrics["syntax_valid"] = result.is_valid
            elif result.validator_name == "schema":
                metrics["schema_valid"] = result.is_valid
            elif result.validator_name == "properties":
                metrics["properties_valid"] = result.is_valid
            elif result.validator_name == "execution":
                metrics["execution_valid"] = result.is_valid

        # KG Valid = syntax + schema + properties all pass
        metrics["kg_valid"] = (
            metrics["syntax_valid"]
            and metrics["schema_valid"]
            and metrics["properties_valid"]
        )

        # If all structural validations passed, check Pass@1
        if metrics["kg_valid"] and metrics["execution_valid"]:
            try:
                result_match = self.result_validator.validate(
                    generated_query, ground_truth_query
                )
                metrics["pass_at_1"] = result_match.is_valid
            except Exception:
                metrics["pass_at_1"] = False

        return all_valid, results, metrics


def create_validation_pipeline(driver: Driver) -> ValidationPipeline:
    """
    Factory function to create a validation pipeline.

    Args:
        driver: Neo4j database driver

    Returns:
        Configured ValidationPipeline instance
    """
    return ValidationPipeline(
        driver=driver,
        stop_on_first_error=True,
        check_empty_results=True,
    )
