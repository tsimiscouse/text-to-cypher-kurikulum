"""
LLM Provider Configuration for Ollama (Local LLM).
Uses Qwen 2.5 Coder model running locally via Ollama.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5-coder:3b"  # Local model

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

DEFAULT_TEMPERATURE = 0.0
REFINEMENT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOP_P = 1.0


@dataclass
class LLMConfigOllama:
    """
    LLM configuration container for Ollama (Local).
    """
    provider: str = "ollama"
    api_key: str = "ollama"  # Ollama doesn't need API key
    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    refinement_temperature: float = REFINEMENT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = DEFAULT_TOP_P

    def validate(self) -> bool:
        """Validate configuration by checking if Ollama is running."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            raise ValueError(
                "Ollama is not running. Please start Ollama with: ollama serve"
            )

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

def validate_ollama_config():
    """Validate that Ollama is running and model is available."""
    import requests

    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise ValueError("Ollama is not responding")

        # Check if model is available
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        if OLLAMA_MODEL not in model_names and f"{OLLAMA_MODEL}:latest" not in model_names:
            available = ", ".join(model_names) if model_names else "none"
            raise ValueError(
                f"Model '{OLLAMA_MODEL}' not found. "
                f"Available models: {available}. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )

        return True

    except requests.exceptions.ConnectionError:
        raise ValueError(
            "Cannot connect to Ollama. Please start Ollama with: ollama serve"
        )


# =============================================================================
# LLM CONFIG FACTORY
# =============================================================================

def get_ollama_config():
    """Get the Ollama LLM configuration dictionary."""
    validate_ollama_config()

    return {
        "api_key": "ollama",
        "base_url": OLLAMA_BASE_URL,
        "model": OLLAMA_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
    }
