"""
Syntax Validator for Cypher queries.

Wraps CyVer SyntaxValidator to validate Cypher syntax.
"""
import logging
from typing import Tuple, Dict, Any
from neo4j import Driver

from core.agent_state import ValidationResult, ErrorType

logger = logging.getLogger(__name__)


class SyntaxValidatorWrapper:
    """
    Wrapper for CyVer SyntaxValidator.

    Validates that a Cypher query has correct syntax.
    """

    def __init__(self, driver: Driver):
        """
        Initialize the syntax validator.

        Args:
            driver: Neo4j database driver
        """
        self.driver = driver

    def validate(self, query: str) -> ValidationResult:
        """
        Validate Cypher syntax.

        Args:
            query: Cypher query to validate

        Returns:
            ValidationResult with is_valid and error details
        """
        try:
            from CyVer import SyntaxValidator

            validator = SyntaxValidator(self.driver)
            is_valid, metadata = validator.validate(query)

            if is_valid:
                return ValidationResult(
                    validator_name="syntax",
                    is_valid=True,
                    error_type=None,
                    error_message=None,
                    metadata={"raw_metadata": metadata},
                )
            else:
                return ValidationResult(
                    validator_name="syntax",
                    is_valid=False,
                    error_type=ErrorType.SYNTAX_ERROR,
                    error_message=self._format_error_message(metadata),
                    metadata={"raw_metadata": metadata},
                )

        except ImportError:
            logger.warning("CyVer not available, using fallback syntax validation")
            return self._fallback_validate(query)

        except Exception as e:
            logger.error(f"Syntax validation error: {str(e)}")
            return ValidationResult(
                validator_name="syntax",
                is_valid=False,
                error_type=ErrorType.SYNTAX_ERROR,
                error_message=str(e),
                metadata={},
            )

    def _fallback_validate(self, query: str) -> ValidationResult:
        """
        Fallback syntax validation using Neo4j EXPLAIN.

        Uses EXPLAIN to check if the query can be parsed without executing.
        """
        try:
            with self.driver.session() as session:
                # Use EXPLAIN to validate without executing
                session.run(f"EXPLAIN {query}")

            return ValidationResult(
                validator_name="syntax",
                is_valid=True,
                error_type=None,
                error_message=None,
                metadata={"method": "neo4j_explain"},
            )

        except Exception as e:
            error_msg = str(e)
            return ValidationResult(
                validator_name="syntax",
                is_valid=False,
                error_type=ErrorType.SYNTAX_ERROR,
                error_message=error_msg,
                metadata={"method": "neo4j_explain"},
            )

    def _format_error_message(self, metadata: Any) -> str:
        """Format the error message from CyVer metadata."""
        if isinstance(metadata, dict):
            return str(metadata.get("error", metadata))
        return str(metadata)
