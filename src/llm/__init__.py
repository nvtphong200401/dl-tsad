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
from .range_prompt_builder import (
    RANGE_DETECTION_SYSTEM_PROMPT,
    build_range_detection_prompt,
    build_evidence_summary,
)
from .range_output_parser import (
    parse_range_output,
    ranges_to_point_scores,
)

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
    'RANGE_DETECTION_SYSTEM_PROMPT',
    'build_range_detection_prompt',
    'build_evidence_summary',
    'parse_range_output',
    'ranges_to_point_scores',
]
