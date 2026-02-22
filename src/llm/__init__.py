"""LLM Reasoning Layer for Anomaly Detection

Provides LLM-based scoring that replaces heuristic evidence aggregation
with intelligent reasoning. Supports Azure OpenAI, Gemini, and Claude.
"""

from .backends import (
    LLMBackend,
    AzureOpenAIBackend,
    GeminiBackend,
    ClaudeBackend,
    create_backend,
)
from .llm_agent import LLMAnomalyAgent
from .prompt_builder import SYSTEM_PROMPT, build_batch_prompt, build_single_prompt
from .output_parser import parse_llm_output, extract_window_confidence

__all__ = [
    'LLMBackend',
    'AzureOpenAIBackend',
    'GeminiBackend',
    'ClaudeBackend',
    'create_backend',
    'LLMAnomalyAgent',
    'SYSTEM_PROMPT',
    'build_batch_prompt',
    'build_single_prompt',
    'parse_llm_output',
    'extract_window_confidence',
]
