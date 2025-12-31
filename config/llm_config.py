"""
LLM Provider Configuration for the Agentic Text2Cypher pipeline.
Uses OpenRouter API for Qwen 2.5 Coder 32B Instruct model.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# OPENROUTER API CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "qwen/qwen-2.5-coder-32b-instruct"

# Legacy Groq config (backup)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3-32b"

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

DEFAULT_TEMPERATURE = 0.0
REFINEMENT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 1.0


@dataclass
class LLMConfig:
    """
    LLM configuration container.
    """
    provider: str = "openrouter"
    api_key: str = ""  # Will be set in __post_init__
    base_url: str = OPENROUTER_BASE_URL
    model: str = OPENROUTER_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    refinement_temperature: float = REFINEMENT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = DEFAULT_TOP_P

    def __post_init__(self):
        """Load API key fresh from environment if not provided."""
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required. Check your .env file.")
        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for API calls."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that all required configuration is present."""
    missing = []

    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please check your .env file."
        )

    return True

# =============================================================================
# LLM CLIENT FACTORY
# =============================================================================

def get_llm_config():
    """Get the LLM configuration dictionary."""
    validate_config()

    return {
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
        "model": OPENROUTER_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
    }
