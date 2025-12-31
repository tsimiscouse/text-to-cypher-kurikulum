"""
Prompt Manager for the Agentic Text2Cypher pipeline.

Handles loading and formatting of prompt templates and schema formats.
"""
import logging
from typing import Dict, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Available prompt engineering techniques."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "cot"


class SchemaFormat(str, Enum):
    """Available schema representation formats."""
    FULL_SCHEMA = "full_schema"
    NODES_PATHS = "nodes_paths"
    ONLY_PATHS = "only_paths"


class PromptManager:
    """
    Manages prompt templates and schema formats.

    Provides methods to load and format prompts for different configurations.
    """

    # Mapping of prompt types to template files
    PROMPT_FILES = {
        PromptType.ZERO_SHOT: "zero_prompt_template.txt",
        PromptType.FEW_SHOT: "few_prompt_template.txt",
        PromptType.CHAIN_OF_THOUGHT: "cot_prompt_template.txt",
    }

    # Mapping of schema formats to files
    SCHEMA_FILES = {
        SchemaFormat.FULL_SCHEMA: "full_schema.txt",
        SchemaFormat.NODES_PATHS: "nodes_paths.txt",
        SchemaFormat.ONLY_PATHS: "only_paths.txt",
    }

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        schemas_dir: Optional[str] = None
    ):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Directory containing prompt templates
            schemas_dir: Directory containing schema formats
        """
        base_dir = Path(__file__).parent.parent

        if prompts_dir is None:
            prompts_dir = base_dir / "prompts" / "templates"
        if schemas_dir is None:
            schemas_dir = base_dir / "schemas" / "formats"

        self.prompts_dir = Path(prompts_dir)
        self.schemas_dir = Path(schemas_dir)

        # Cache for loaded templates
        self._prompt_cache: Dict[PromptType, str] = {}
        self._schema_cache: Dict[SchemaFormat, str] = {}
        self._refinement_template: Optional[str] = None

    def load_prompt_template(self, prompt_type: PromptType) -> str:
        """
        Load a prompt template.

        Args:
            prompt_type: The type of prompt to load

        Returns:
            The prompt template string
        """
        if prompt_type in self._prompt_cache:
            return self._prompt_cache[prompt_type]

        filename = self.PROMPT_FILES.get(prompt_type)
        if not filename:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        filepath = self.prompts_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Prompt template not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            template = f.read()

        self._prompt_cache[prompt_type] = template
        logger.debug(f"Loaded prompt template: {prompt_type.value}")

        return template

    def load_schema(self, schema_format: SchemaFormat) -> str:
        """
        Load a schema format.

        Args:
            schema_format: The schema format to load

        Returns:
            The schema content string
        """
        if schema_format in self._schema_cache:
            return self._schema_cache[schema_format]

        filename = self.SCHEMA_FILES.get(schema_format)
        if not filename:
            raise ValueError(f"Unknown schema format: {schema_format}")

        filepath = self.schemas_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Schema file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            schema = f.read()

        self._schema_cache[schema_format] = schema
        logger.debug(f"Loaded schema format: {schema_format.value}")

        return schema

    def load_refinement_template(self) -> str:
        """
        Load the refinement prompt template.

        Returns:
            The refinement template string
        """
        if self._refinement_template is not None:
            return self._refinement_template

        filepath = self.prompts_dir / "refinement_template.txt"

        if not filepath.exists():
            # Return default refinement template if file doesn't exist
            self._refinement_template = self._get_default_refinement_template()
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                self._refinement_template = f.read()

        return self._refinement_template

    def _get_default_refinement_template(self) -> str:
        """Get default refinement template."""
        return """Anda adalah asisten yang membantu memperbaiki kueri Cypher.

{feedback}

### Instruksi
Berdasarkan umpan balik di atas, perbaiki kueri Cypher Anda.
Pastikan kueri baru:
1. Memperbaiki semua kesalahan yang disebutkan
2. Mengikuti skema yang diberikan dengan tepat
3. Menjawab pertanyaan asli dengan benar

### Pertanyaan Asli
{question}

### Skema Database
{schema}

Berikan kueri Cypher yang telah diperbaiki:"""

    def format_initial_prompt(
        self,
        prompt_type: PromptType,
        schema_format: SchemaFormat,
        question: str
    ) -> str:
        """
        Format an initial prompt with schema and question.

        Args:
            prompt_type: The prompt engineering technique to use
            schema_format: The schema representation format to use
            question: The natural language question

        Returns:
            Formatted prompt string
        """
        template = self.load_prompt_template(prompt_type)
        schema = self.load_schema(schema_format)

        # The templates use {schema} and {question} placeholders
        formatted = template.format(schema=schema, question=question)

        return formatted

    def format_refinement_prompt(
        self,
        question: str,
        schema_format: SchemaFormat,
        feedback: str
    ) -> str:
        """
        Format a refinement prompt with feedback.

        Args:
            question: The original question
            schema_format: The schema format to include
            feedback: The structured feedback from validation

        Returns:
            Formatted refinement prompt
        """
        template = self.load_refinement_template()
        schema = self.load_schema(schema_format)

        formatted = template.format(
            question=question,
            schema=schema,
            feedback=feedback
        )

        return formatted

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.

        Returns:
            System prompt string
        """
        return """Anda adalah asisten ahli dalam mengonversi pertanyaan bahasa alami ke kueri Cypher untuk database Neo4j.

Tugas Anda:
1. Analisis pertanyaan yang diberikan
2. Identifikasi entitas dan relasi yang relevan dari skema
3. Hasilkan kueri Cypher yang valid dan efisien

Aturan:
- Gunakan HANYA label node dan tipe relasi yang ada dalam skema
- Gunakan HANYA properti yang didefinisikan untuk setiap jenis node
- Pastikan sintaks Cypher benar
- Berikan kueri yang menjawab pertanyaan secara tepat

Format output:
- Berikan proses berpikir Anda dalam tag <think>...</think>
- Berikan kueri Cypher akhir dalam tag <cypher>...</cypher>"""

    def get_configuration_name(
        self,
        prompt_type: PromptType,
        schema_format: SchemaFormat
    ) -> str:
        """
        Get a descriptive name for a configuration.

        Args:
            prompt_type: The prompt type
            schema_format: The schema format

        Returns:
            Configuration name string
        """
        prompt_names = {
            PromptType.ZERO_SHOT: "Zero-Shot",
            PromptType.FEW_SHOT: "Few-Shot",
            PromptType.CHAIN_OF_THOUGHT: "CoT",
        }

        schema_names = {
            SchemaFormat.FULL_SCHEMA: "Full",
            SchemaFormat.NODES_PATHS: "Nodes+Paths",
            SchemaFormat.ONLY_PATHS: "Paths",
        }

        return f"{prompt_names[prompt_type]}_{schema_names[schema_format]}"


def create_prompt_manager(
    prompts_dir: Optional[str] = None,
    schemas_dir: Optional[str] = None
) -> PromptManager:
    """
    Factory function to create a PromptManager.

    Args:
        prompts_dir: Optional directory for prompt templates
        schemas_dir: Optional directory for schema formats

    Returns:
        Configured PromptManager instance
    """
    return PromptManager(prompts_dir, schemas_dir)
