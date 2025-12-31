"""
LLM Client for the Agentic Text2Cypher pipeline.

This module provides a wrapper for the Groq API to generate Cypher queries
using the Qwen 2.5 Coder 32B Instruct model.
"""
import re
import time
import logging
from typing import Dict, Optional, Any
from openai import OpenAI

from config.llm_config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM Client wrapper for Groq API.

    Handles API calls, response parsing, and error handling for
    the Qwen 2.5 Coder 32B Instruct model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Groq API key (defaults to env variable)
            base_url: Groq API base URL
            model: Model identifier
        """
        self.api_key = api_key or GROQ_API_KEY
        self.base_url = base_url or GROQ_BASE_URL
        self.model = model or GROQ_MODEL

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Please set it in your .env file."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.info(f"LLM Client initialized with model: {self.model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System prompt with instructions and schema
            user_prompt: User prompt with the question
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter

        Returns:
            Dict with 'cypher' (extracted query) and 'reasoning' (if present)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            raw_content = response.choices[0].message.content
            return self._parse_response(raw_content)

        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise

    def _parse_response(self, raw_content: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract Cypher query and reasoning.

        The model may return responses in different formats:
        1. With <think>...</think> tags containing reasoning
        2. Plain Cypher query
        3. With markdown code blocks

        Args:
            raw_content: Raw response from the LLM

        Returns:
            Dict with 'cypher' and 'reasoning' keys
        """
        reasoning = None
        cypher = raw_content

        # Extract reasoning from <think> tags if present
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, raw_content, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            # Remove the thinking section from the response
            cypher = re.sub(think_pattern, "", raw_content, flags=re.DOTALL).strip()

        # Remove markdown code blocks if present
        cypher = self._remove_code_blocks(cypher)

        # Clean up the query
        cypher = self._clean_query(cypher)

        return {
            "cypher": cypher,
            "reasoning": reasoning,
            "raw_response": raw_content,
        }

    def _remove_code_blocks(self, text: str) -> str:
        """Remove markdown code blocks from text."""
        # Remove ```cypher ... ``` or ```sql ... ``` or ``` ... ```
        patterns = [
            r"```cypher\s*(.*?)\s*```",
            r"```sql\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return text

    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize a Cypher query.

        - Remove leading/trailing whitespace
        - Remove extra newlines
        - Ensure single-line format if possible
        """
        # Remove leading/trailing whitespace
        query = query.strip()

        # Remove common prefixes the model might add
        prefixes_to_remove = [
            "Here is the Cypher query:",
            "The Cypher query is:",
            "Cypher:",
            "Query:",
            "Answer:",
        ]

        for prefix in prefixes_to_remove:
            if query.lower().startswith(prefix.lower()):
                query = query[len(prefix) :].strip()

        # Collapse multiple whitespace/newlines to single space
        # (keeping the query on one line as required)
        query = " ".join(query.split())

        return query

    def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate with automatic retry on failure.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional arguments for generate()

        Returns:
            Dict with 'cypher' and 'reasoning'
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.generate(system_prompt, user_prompt, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        raise last_error


def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from file."""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def load_schema(schema_path: str) -> str:
    """Load a schema format from file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt(template_content: str, schema_content: str) -> str:
    """
    Build the system prompt by injecting schema into template.

    Args:
        template_content: Prompt template with {schema} placeholder
        schema_content: Schema format content

    Returns:
        Complete system prompt
    """
    return template_content.format(schema=schema_content)
