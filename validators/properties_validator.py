"""
Properties Validator for Cypher queries.

Wraps CyVer PropertiesValidator to validate property names.
"""
import logging
from typing import Tuple, Dict, Any, List, Set
from neo4j import Driver

from core.agent_state import ValidationResult, ErrorType

logger = logging.getLogger(__name__)


class PropertiesValidatorWrapper:
    """
    Wrapper for CyVer PropertiesValidator.

    Validates that a Cypher query uses valid property names
    for each node type.
    """

    def __init__(self, driver: Driver):
        """
        Initialize the properties validator.

        Args:
            driver: Neo4j database driver
        """
        self.driver = driver

        # Valid properties for each node type in the curriculum KG
        self.valid_properties = {
            "MK": {
                "kode",
                "nama",
                "sks",
                "durasi",
                "tipe",
                "klasifikasi",
                "semester",
                "jalur",
            },
            "LO": {"nama", "deskripsi", "MK"},
            "LG": {"nama", "deskripsi", "MK"},
            "topic": {"nama", "deskripsi", "MK"},
            "SO": {"kode", "deskripsi"},
        }

        # All valid properties across all node types
        self.all_valid_properties: Set[str] = set()
        for props in self.valid_properties.values():
            self.all_valid_properties.update(props)

    def validate(self, query: str, strict: bool = False) -> ValidationResult:
        """
        Validate property names in the query.

        Args:
            query: Cypher query to validate
            strict: If True, validate property-node associations

        Returns:
            ValidationResult with is_valid and error details
        """
        try:
            from CyVer import PropertiesValidator

            validator = PropertiesValidator(self.driver)
            score, metadata = validator.validate(query, strict=strict)

            # Extract variables and labels
            variables, labels = validator.extract(query)

            is_valid = score == 1.0

            if is_valid:
                return ValidationResult(
                    validator_name="properties",
                    is_valid=True,
                    error_type=None,
                    error_message=None,
                    metadata={
                        "score": score,
                        "extracted_variables": variables,
                        "extracted_labels": labels,
                        "raw_metadata": metadata,
                    },
                )
            else:
                return ValidationResult(
                    validator_name="properties",
                    is_valid=False,
                    error_type=ErrorType.PROPERTIES_ERROR,
                    error_message=self._format_error_message(metadata, variables),
                    metadata={
                        "score": score,
                        "extracted_variables": variables,
                        "extracted_labels": labels,
                        "raw_metadata": metadata,
                    },
                )

        except ImportError:
            logger.warning("CyVer not available, using fallback properties validation")
            return self._fallback_validate(query)

        except Exception as e:
            logger.error(f"Properties validation error: {str(e)}")
            return ValidationResult(
                validator_name="properties",
                is_valid=False,
                error_type=ErrorType.PROPERTIES_ERROR,
                error_message=str(e),
                metadata={},
            )

    def _fallback_validate(self, query: str) -> ValidationResult:
        """
        Fallback properties validation using regex extraction.

        Extracts property accesses and checks against known valid properties.
        """
        import re

        # Extract property accesses (pattern: variable.property or {property: value})
        # Pattern 1: n.property
        dot_pattern = r"\.(\w+)"
        dot_properties = set(re.findall(dot_pattern, query))

        # Pattern 2: {property: value} or {property: "value"}
        brace_pattern = r"\{[^}]*?(\w+)\s*:"
        brace_properties = set(re.findall(brace_pattern, query))

        all_found_properties = dot_properties | brace_properties

        # Filter out common Cypher keywords that might be caught
        keywords_to_ignore = {
            "count",
            "sum",
            "avg",
            "collect",
            "size",
            "length",
            "type",
            "id",
            "labels",
            "keys",
            "properties",
            "nodes",
            "relationships",
        }
        all_found_properties -= keywords_to_ignore

        # Check for invalid properties
        invalid_properties = all_found_properties - self.all_valid_properties

        is_valid = len(invalid_properties) == 0

        if is_valid:
            return ValidationResult(
                validator_name="properties",
                is_valid=True,
                error_type=None,
                error_message=None,
                metadata={
                    "method": "regex_fallback",
                    "extracted_properties": list(all_found_properties),
                },
            )
        else:
            return ValidationResult(
                validator_name="properties",
                is_valid=False,
                error_type=ErrorType.PROPERTIES_ERROR,
                error_message=f"Invalid properties: {invalid_properties}",
                metadata={
                    "method": "regex_fallback",
                    "extracted_properties": list(all_found_properties),
                    "invalid_properties": list(invalid_properties),
                },
            )

    def _format_error_message(self, metadata: Any, variables: List[str]) -> str:
        """Format the error message with extracted variables."""
        parts = []
        if metadata:
            parts.append(str(metadata))
        if variables:
            parts.append(f"Variables used: {variables}")
        return "; ".join(parts)
