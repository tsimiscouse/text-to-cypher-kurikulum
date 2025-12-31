"""
Global settings and configuration for the Agentic Text2Cypher pipeline.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Settings:
    """
    Application settings container.

    Provides a structured way to access configuration values.
    """
    # Agentic Loop
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", 3))
    temperature: float = float(os.getenv("TEMPERATURE", 0.0))
    refinement_temperature: float = 0.1
    max_tokens: int = int(os.getenv("MAX_TOKENS", 512))

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    # Rate Limiting
    rate_limit_delay: float = 3.0
    batch_size: int = 20
    batch_pause: float = 60.0

    # Paths (set in __post_init__)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        self.prompts_dir = self.base_dir / "prompts" / "templates"
        self.schemas_dir = self.base_dir / "schemas" / "formats"

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "ground-truth_refined.csv"

# Prompt and schema paths
PROMPTS_DIR = BASE_DIR / "prompts" / "templates"
SCHEMAS_DIR = BASE_DIR / "schemas" / "formats"
FEEDBACK_TEMPLATES_DIR = BASE_DIR / "feedback" / "templates"

# Results paths
RESULTS_DIR = BASE_DIR / "results"

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATES = {
    "zero": PROMPTS_DIR / "zero_prompt_template.txt",
    "few": PROMPTS_DIR / "few_prompt_template.txt",
    "cot": PROMPTS_DIR / "cot_prompt_template.txt",
}

# =============================================================================
# SCHEMA FORMATS
# =============================================================================

SCHEMA_FORMATS = {
    "full_schema": SCHEMAS_DIR / "full_schema.txt",
    "nodes_paths": SCHEMAS_DIR / "nodes_paths.txt",
    "only_paths": SCHEMAS_DIR / "only_paths.txt",
}

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# 3x3 Factorial Design: 9 configurations
EXPERIMENT_CONFIGS = [
    {"prompt": "zero", "schema": "full_schema", "name": "zero_full_schema"},
    {"prompt": "zero", "schema": "nodes_paths", "name": "zero_nodes_paths"},
    {"prompt": "zero", "schema": "only_paths", "name": "zero_only_paths"},
    {"prompt": "few", "schema": "full_schema", "name": "few_full_schema"},
    {"prompt": "few", "schema": "nodes_paths", "name": "few_nodes_paths"},
    {"prompt": "few", "schema": "only_paths", "name": "few_only_paths"},
    {"prompt": "cot", "schema": "full_schema", "name": "cot_full_schema"},
    {"prompt": "cot", "schema": "nodes_paths", "name": "cot_nodes_paths"},
    {"prompt": "cot", "schema": "only_paths", "name": "cot_only_paths"},
]

# =============================================================================
# AGENTIC LOOP CONFIGURATION
# =============================================================================

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 3))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
REFINEMENT_TEMPERATURE = 0.1  # Slightly higher for refinement attempts
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))

# =============================================================================
# NEO4J CONFIGURATION
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# =============================================================================
# RATE LIMITING
# =============================================================================

RATE_LIMIT_DELAY = 3.0  # Seconds between API calls
BATCH_SIZE = 20  # Questions per batch before longer pause
BATCH_PAUSE = 60  # Seconds to pause between batches

# =============================================================================
# GROUND TRUTH COLUMNS
# =============================================================================

GT_COLUMNS = {
    "reasoning_level": "Tingkat Penalaran",
    "sublevel": "Sublevel",
    "complexity": "Tingkat Kompleksitas",
    "question": "Pertanyaan",
    "ground_truth": "Cypher Query",
}

# =============================================================================
# OUTPUT COLUMNS
# =============================================================================

OUTPUT_COLUMNS = [
    "question_id",
    "question",
    "ground_truth",
    "reasoning_level",
    "sublevel",
    "complexity",
    "final_query",
    "success",
    "total_iterations",
    "first_attempt_success",
    "iteration_history",
]
