"""
Feedback Builder for the Agentic Text2Cypher pipeline.

Constructs structured feedback for the LLM based on validation errors.
The feedback is designed to give the LLM specific, actionable information
to correct its query.
"""
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from core.agent_state import AgentState, Attempt, ValidationResult, ErrorType

logger = logging.getLogger(__name__)


class FeedbackBuilder:
    """
    Constructs structured feedback for the LLM based on validation errors.

    The feedback is designed to give the LLM specific, actionable information
    to correct its query while avoiding prompt injection.
    """

    def __init__(self, schema_content: str):
        """
        Initialize with schema content for reference.

        Args:
            schema_content: The KG schema being used
        """
        self.schema_content = schema_content
        self.templates: Dict[ErrorType, str] = {}
        self._load_templates()

    def _load_templates(self):
        """Load feedback templates from files."""
        template_dir = Path(__file__).parent / "templates"

        template_files = {
            ErrorType.SYNTAX_ERROR: "syntax_error.txt",
            ErrorType.SCHEMA_ERROR: "schema_error.txt",
            ErrorType.PROPERTIES_ERROR: "properties_error.txt",
            ErrorType.EXECUTION_ERROR: "execution_error.txt",
            ErrorType.EMPTY_RESULT: "empty_result.txt",
        }

        for error_type, filename in template_files.items():
            filepath = template_dir / filename
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self.templates[error_type] = f.read()
            except FileNotFoundError:
                logger.warning(f"Template not found: {filepath}")
                self.templates[error_type] = self._get_default_template(error_type)

    def _get_default_template(self, error_type: ErrorType) -> str:
        """Get a default template if file is missing."""
        return f"""**Kesalahan {error_type.value.upper()} Terdeteksi**

Pesan Error: {{error_message}}

Silakan perbaiki kueri Anda."""

    def build_feedback(self, state: AgentState, attempt: Attempt) -> str:
        """
        Build comprehensive feedback for refinement.

        Args:
            state: Current agent state
            attempt: The failed attempt to provide feedback on

        Returns:
            Formatted feedback string for the LLM
        """
        feedback_parts = []

        # Header
        feedback_parts.append(self._build_header(state, attempt))

        # Previous query
        feedback_parts.append(self._build_previous_query_section(attempt))

        # Error details
        feedback_parts.append(self._build_error_section(state, attempt))

        # Correction hints
        hints = self._build_hints_section(attempt)
        if hints:
            feedback_parts.append(hints)

        # Schema reminder (condensed)
        feedback_parts.append(self._build_schema_reminder())

        return "\n\n".join(filter(None, feedback_parts))

    def _build_header(self, state: AgentState, attempt: Attempt) -> str:
        """Build feedback header."""
        return f"""## Koreksi Diperlukan (Iterasi {attempt.iteration + 2}/{state.max_iterations})

Kueri Cypher sebelumnya mengandung kesalahan. Silakan perbaiki berdasarkan umpan balik berikut."""

    def _build_previous_query_section(self, attempt: Attempt) -> str:
        """Show the previous failed query."""
        return f"""### Kueri Sebelumnya (GAGAL)
```cypher
{attempt.generated_query}
```"""

    def _build_error_section(self, state: AgentState, attempt: Attempt) -> str:
        """Build detailed error section."""
        error_details = []

        for validation in attempt.validation_results:
            if not validation.is_valid:
                error_detail = self._format_error(state, validation)
                error_details.append(error_detail)

        if not error_details:
            return "### Detail Kesalahan\nTidak ada detail kesalahan tersedia."

        return f"""### Detail Kesalahan
{chr(10).join(error_details)}"""

    def _format_error(self, state: AgentState, validation: ValidationResult) -> str:
        """Format a single error with specific details."""
        error_type = validation.error_type

        if error_type not in self.templates:
            return f"**Error:** {validation.error_message}"

        template = self.templates[error_type]

        # Build context for template
        context = {
            "error_message": validation.error_message or "Unknown error",
            "question": state.question,
        }

        # Add metadata-specific context
        metadata = validation.metadata
        if error_type == ErrorType.SCHEMA_ERROR:
            context["extracted_nodes"] = metadata.get("extracted_nodes", [])
            context["extracted_relationships"] = metadata.get(
                "extracted_relationships", []
            )
        elif error_type == ErrorType.PROPERTIES_ERROR:
            context["extracted_properties"] = metadata.get(
                "extracted_variables", metadata.get("extracted_properties", [])
            )

        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            return template

    def _build_hints_section(self, attempt: Attempt) -> Optional[str]:
        """Build correction hints based on error type."""
        primary_error = attempt.primary_error

        if not primary_error:
            return None

        hints = {
            ErrorType.SYNTAX_ERROR: [
                "Pastikan semua tanda kurung berpasangan dengan benar",
                "Periksa penggunaan tanda kutip untuk string",
                "Pastikan kata kunci Cypher (MATCH, WHERE, RETURN) dieja dengan benar",
                "Periksa urutan clause: MATCH → WHERE → WITH → RETURN",
            ],
            ErrorType.SCHEMA_ERROR: [
                "Label node yang valid: MK, LO, LG, topic, SO",
                "Tipe relasi yang valid: PREREQUISITE, CAN_PARALLELIZED, PURSUED_IN, PART_OF",
                "Periksa arah relasi sesuai dengan skema",
                "Gunakan label persis seperti dalam skema (case-sensitive)",
            ],
            ErrorType.PROPERTIES_ERROR: [
                "Properti MK: kode, nama, sks, durasi, tipe, klasifikasi, semester, jalur",
                "Properti LO/LG/topic: nama, deskripsi, MK",
                "Properti SO: kode, deskripsi",
                "Periksa ejaan properti (case-sensitive)",
            ],
            ErrorType.EXECUTION_ERROR: [
                "Periksa apakah variabel sudah didefinisikan sebelum digunakan",
                "Pastikan agregasi menggunakan WITH clause jika diperlukan",
                "Gunakan COALESCE untuk handle nilai NULL jika perlu",
            ],
            ErrorType.EMPTY_RESULT: [
                "Periksa nilai filter (case-sensitive untuk string)",
                "Pertimbangkan apakah pola relasi sudah benar",
                "Coba gunakan OPTIONAL MATCH jika data mungkin tidak ada",
                "Periksa kembali pertanyaan asli",
            ],
        }

        if primary_error in hints:
            hint_list = "\n".join(f"- {h}" for h in hints[primary_error])
            return f"""### Petunjuk Perbaikan
{hint_list}"""

        return None

    def _build_schema_reminder(self) -> str:
        """Build condensed schema reminder."""
        # Use a condensed version of the schema
        condensed_schema = """Node Labels: MK, LO, LG, topic, SO
Relationships: PREREQUISITE, CAN_PARALLELIZED, PURSUED_IN, PART_OF"""

        return f"""### Referensi Skema (Ringkas)
{condensed_schema}"""


def create_feedback_builder(schema_content: str) -> FeedbackBuilder:
    """
    Factory function to create a FeedbackBuilder.

    Args:
        schema_content: The KG schema content

    Returns:
        Configured FeedbackBuilder instance
    """
    return FeedbackBuilder(schema_content)
