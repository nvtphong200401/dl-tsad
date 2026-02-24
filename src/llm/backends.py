"""LLM Backend Wrappers

Unified interface for Azure OpenAI, Google Gemini, and Anthropic Claude.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

# Load .env file from project root
from dotenv import load_dotenv
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


class LLMBackend(ABC):
    """Base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """Generate completion from LLM.

        Args:
            system_prompt: System role instructions
            user_prompt: User message with evidence
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Raw text response from LLM
        """
        pass

    def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """Generate with exponential backoff on failure."""
        for attempt in range(max_retries):
            try:
                return self.generate(system_prompt, user_prompt, temperature)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  LLM API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)


class AzureOpenAIBackend(LLMBackend):
    """Azure OpenAI Service backend.

    Requires:
        - AZURE_OPENAI_ENDPOINT: https://{resource}.openai.azure.com/
        - AZURE_OPENAI_API_KEY: API key
        - AZURE_OPENAI_DEPLOYMENT: Deployment name (e.g. gpt-4o)
    """

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: str = "2024-12-01-preview"
    ):
        from openai import AzureOpenAI

        self.deployment = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        kwargs = dict(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class GeminiBackend(LLMBackend):
    """Google Gemini backend.

    Requires:
        - GOOGLE_API_KEY: API key
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash"
    ):
        import google.generativeai as genai

        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(
            model,
            generation_config={"response_mime_type": "application/json"}
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        # Gemini: prepend system prompt to user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        response = self.model.generate_content(
            full_prompt,
            generation_config=gen_config
        )
        return response.text


class ClaudeBackend(LLMBackend):
    """Anthropic Claude backend.

    Requires:
        - ANTHROPIC_API_KEY: API key
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-latest"
    ):
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        kwargs = dict(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4096
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.messages.create(**kwargs)
        return response.content[0].text


def create_backend(backend_type: str, **kwargs) -> LLMBackend:
    """Factory function to create LLM backend.

    Args:
        backend_type: "azure_openai", "gemini", or "claude"
        **kwargs: Backend-specific configuration

    Returns:
        Configured LLMBackend instance
    """
    backends = {
        "azure_openai": AzureOpenAIBackend,
        "gemini": GeminiBackend,
        "claude": ClaudeBackend,
    }
    if backend_type not in backends:
        raise ValueError(f"Unknown backend: {backend_type}. Choose from: {list(backends.keys())}")
    return backends[backend_type](**kwargs)
