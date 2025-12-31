"""
Schema Validator for Cypher queries.

Wraps CyVer SchemaValidator to validate node labels and relationships.
"""
import logging
from typing import Tuple, Dict, Any, List
from neo4j import Driver

from core.agent_state import ValidationResult, ErrorType

logger = logging.getLogger(__name__)


class SchemaValidatorWrapper:
    """
    Wrapper for CyVer SchemaValidator.

    Validates that a Cypher query uses valid node labels and
    relationship types according to the knowledge graph schema.
    """

    def __init__(self, driver: Driver):
        """
        Initialize the schema validator.

        Args:
            driver: Neo4j database driver
        """
        self.driver = driver

        # Valid schema elements for the curriculum KG
        self.valid_node_labels = {"MK", "LO", "LG", "topic", "SO"}
        self.valid_relationship_types = {
            "PREREQUISITE",
            "CAN_PARALLELIZED",
            "PURSUED_IN",
            "PART_OF",
        }

    def validate(self, query: str) -> ValidationResult:
        """
        Validate schema elements in the query.

        Args:
            query: Cypher query to validate

        Returns:
            ValidationResult with is_valid and error details
        """
        try:
            from CyVer import SchemaValidator

            validator = SchemaValidator(self.driver)
            score, metadata = validator.validate(query)

            # Extract schema elements
            nodes, relationships, paths = validator.extract(query)

            is_valid = score == 1.0

            if is_valid:
                return ValidationResult(
                    validator_name="schema",
                    is_valid=True,
                    error_type=None,
                    error_message=None,
                    metadata={
                        "score": score,
                        "extracted_nodes": nodes,
                        "extracted_relationships": relationships,
                        "extracted_paths": paths,
                        "raw_metadata": metadata,
                    },
                )
            else:
                return ValidationResult(
                    validator_name="schema",
                    is_valid=False,
                    error_type=ErrorType.SCHEMA_ERROR,
                    error_message=self._format_error_message(
                        metadata, nodes, relationships
                    ),
                    metadata={
                        "score": score,
                        "extracted_nodes": nodes,
                        "extracted_relationships": relationships,
                        "extracted_paths": paths,
                        "raw_metadata": metadata,
                    },
                )

        except ImportError:
            logger.warning("CyVer not available, using fallback schema validation")
            return self._fallback_validate(query)

        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return ValidationResult(
                validator_name="schema",
                is_valid=False,
                error_type=ErrorType.SCHEMA_ERROR,
                error_message=str(e),
                metadata={},
            )

    def _fallback_validate(self, query: str) -> ValidationResult:
        """
        Fallback schema validation using regex extraction.

        Extracts node labels and relationship types and checks against
        known valid values.
        """
        import re

        # Extract node labels (pattern: :LabelName or (variable:LabelName))
        label_pattern = r":(\w+)"
        found_labels = set(re.findall(label_pattern, query))

        # Extract relationship types (pattern: [:TYPE] or -[:TYPE]->)
        rel_pattern = r"\[:(\w+)\]"
        found_relationships = set(re.findall(rel_pattern, query))

        # Check for invalid labels
        invalid_labels = found_labels - self.valid_node_labels
        invalid_rels = found_relationships - self.valid_relationship_types

        is_valid = len(invalid_labels) == 0 and len(invalid_rels) == 0

        if is_valid:
            return ValidationResult(
                validator_name="schema",
                is_valid=True,
                error_type=None,
                error_message=None,
                metadata={
                    "method": "regex_fallback",
                    "extracted_nodes": list(found_labels),
                    "extracted_relationships": list(found_relationships),
                },
            )
        else:
            error_parts = []
            if invalid_labels:
                error_parts.append(f"Invalid node labels: {invalid_labels}")
            if invalid_rels:
                error_parts.append(f"Invalid relationship types: {invalid_rels}")

            return ValidationResult(
                validator_name="schema",
                is_valid=False,
                error_type=ErrorType.SCHEMA_ERROR,
                error_message="; ".join(error_parts),
                metadata={
                    "method": "regex_fallback",
                    "extracted_nodes": list(found_labels),
                    "extracted_relationships": list(found_relationships),
                    "invalid_labels": list(invalid_labels),
                    "invalid_relationships": list(invalid_rels),
                },
            )

    def _format_error_message(
        self, metadata: Any, nodes: List[str], relationships: List[str]
    ) -> str:
        """Format the error message with extracted elements."""
        parts = []
        if metadata:
            parts.append(str(metadata))
        parts.append(f"Nodes used: {nodes}")
        parts.append(f"Relationships used: {relationships}")
        return "; ".join(parts)
