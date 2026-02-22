"""Tests for LLM reasoning layer.

Tests prompt builder, output parser, and agent logic WITHOUT requiring API keys.
"""

import pytest
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm.prompt_builder import (
    SYSTEM_PROMPT,
    build_batch_prompt,
    build_single_prompt,
    format_forecast_evidence,
    format_statistical_tests,
    format_distribution_evidence,
    format_pattern_evidence,
    format_time_series,
)
from src.llm.output_parser import (
    parse_llm_output,
    validate_result,
    extract_window_confidence,
)
from src.llm.backends import LLMBackend


# ============================================================
# Mock Backend for Testing
# ============================================================

class MockBackend(LLMBackend):
    """Mock backend that returns pre-configured responses."""

    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls = []

    def generate(self, system_prompt, user_prompt, temperature=0.0):
        self.calls.append({
            "system": system_prompt,
            "user": user_prompt,
            "temperature": temperature
        })
        return self.response


# ============================================================
# Prompt Builder Tests
# ============================================================

class TestPromptBuilder:

    def test_system_prompt_exists(self):
        assert len(SYSTEM_PROMPT) > 100
        assert "anomaly" in SYSTEM_PROMPT.lower()
        assert "JSON" in SYSTEM_PROMPT

    def test_format_time_series_short(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = format_time_series(values)
        assert "5 points" in result
        assert "1.000" in result

    def test_format_time_series_long(self):
        values = np.random.randn(200)
        result = format_time_series(values, max_points=50)
        assert "First 10" in result
        assert "Last 10" in result

    def test_format_forecast_evidence(self):
        evidence = {
            'mae': 2.5,
            'mse': 8.0,
            'mape': 15.3,
            'violation_ratio': 0.25,
            'extreme_violation': True,
            'mean_surprise': 3.5,
        }
        result = format_forecast_evidence(evidence)
        assert "MAE" in result
        assert "2.5" in result
        assert "Extreme Quantile Violation: YES" in result
        assert "Surprise Score" in result

    def test_format_forecast_evidence_empty(self):
        result = format_forecast_evidence({})
        assert result == ""

    def test_format_statistical_tests(self):
        evidence = {
            'max_abs_z_score': 4.2,
            'extreme_z_count': 3,
            'grubbs_is_outlier': True,
            'grubbs_outlier_index': 25,
            'max_cusum': 12.5,
            'cusum_has_change_point': True,
        }
        result = format_statistical_tests(evidence)
        assert "4.20" in result
        assert "ANOMALOUS" in result
        assert "OUTLIER detected at position 25" in result
        assert "Change point detected" in result

    def test_format_distribution_evidence(self):
        evidence = {
            'kl_divergence': 1.5,
            'normalized_wasserstein': 2.3,
        }
        result = format_distribution_evidence(evidence)
        assert "KL Divergence" in result
        assert "HIGH" in result
        assert "Wasserstein" in result
        assert "SIGNIFICANT" in result

    def test_format_pattern_evidence(self):
        evidence = {
            'volatility_ratio': 6.0,
            'max_acf_diff': 0.35,
            'period_changed': True,
            'slope_diff': 0.15,
            'trend_break': True,
        }
        result = format_pattern_evidence(evidence)
        assert "EXTREME" in result
        assert "Periodicity: CHANGED" in result
        assert "Trend Break: YES" in result

    def test_build_single_prompt(self):
        window = np.random.randn(50)
        evidence = {'mae': 1.5, 'max_abs_z_score': 2.0}
        result = build_single_prompt(window, evidence, window_index=5)
        assert "Window 5" in result
        assert "MAE" in result

    def test_build_batch_prompt(self):
        windows = [np.random.randn(30) for _ in range(3)]
        evidence_list = [
            {'mae': 1.0, 'max_abs_z_score': 1.5},
            {'mae': 5.0, 'max_abs_z_score': 4.2},
            {'mae': 0.5, 'max_abs_z_score': 0.8},
        ]
        result = build_batch_prompt(windows, evidence_list, [0, 1, 2])
        assert "3 windows" in result
        assert "Window 0" in result
        assert "Window 1" in result
        assert "Window 2" in result


# ============================================================
# Output Parser Tests
# ============================================================

class TestOutputParser:

    def test_parse_valid_json(self):
        raw = json.dumps({
            "windows": [
                {
                    "window_index": 0,
                    "is_anomaly": True,
                    "confidence": 0.85,
                    "reasoning": "High z-score",
                    "evidence_cited": ["z_score"]
                }
            ]
        })
        result = parse_llm_output(raw)
        assert result["parse_error"] is None
        assert len(result["windows"]) == 1
        assert result["windows"][0]["confidence"] == 0.85
        assert result["windows"][0]["is_anomaly"] is True

    def test_parse_json_in_code_block(self):
        raw = '```json\n{"windows": [{"window_index": 0, "is_anomaly": false, "confidence": 0.1}]}\n```'
        result = parse_llm_output(raw)
        assert result["parse_error"] is None
        assert len(result["windows"]) == 1
        assert result["windows"][0]["is_anomaly"] is False

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is my analysis:\n{"windows": [{"window_index": 0, "confidence": 0.9, "is_anomaly": true}]}\nDone.'
        result = parse_llm_output(raw)
        assert result["parse_error"] is None
        assert result["windows"][0]["confidence"] == 0.9

    def test_parse_anomalies_format(self):
        """Handle the alternative 'anomalies' format from spec."""
        raw = json.dumps({
            "anomalies": [
                {
                    "start_index": 42,
                    "end_index": 48,
                    "confidence": 0.92,
                    "reasoning": "High z-score",
                    "evidence_cited": ["z_score"]
                }
            ],
            "overall_assessment": "Anomaly detected"
        })
        result = parse_llm_output(raw)
        assert result["parse_error"] is None
        assert len(result["windows"]) == 1
        assert result["windows"][0]["confidence"] == 0.92

    def test_parse_empty_response(self):
        result = parse_llm_output("")
        assert result["parse_error"] is not None
        assert len(result["windows"]) == 0

    def test_parse_invalid_json(self):
        result = parse_llm_output("This is not JSON at all")
        assert result["parse_error"] is not None

    def test_confidence_clamped(self):
        raw = json.dumps({
            "windows": [{"window_index": 0, "confidence": 1.5, "is_anomaly": True}]
        })
        result = parse_llm_output(raw)
        assert result["windows"][0]["confidence"] == 1.0

    def test_extract_window_confidence(self):
        parsed = {
            "windows": [
                {"window_index": 0, "confidence": 0.8},
                {"window_index": 1, "confidence": 0.2},
                {"window_index": 2, "confidence": 0.95},
            ]
        }
        confidence = extract_window_confidence(parsed, [0, 1, 2, 3])
        assert confidence[0] == 0.8
        assert confidence[1] == 0.2
        assert confidence[2] == 0.95
        assert confidence[3] == 0.0  # Missing → default 0.0


# ============================================================
# LLM Agent Tests (with mock backend)
# ============================================================

class TestLLMAgent:

    def test_agent_single_analysis(self):
        from src.llm.llm_agent import LLMAnomalyAgent

        mock_response = json.dumps({
            "windows": [
                {"window_index": 0, "is_anomaly": True, "confidence": 0.9,
                 "reasoning": "Test", "evidence_cited": ["mae"]}
            ]
        })
        backend = MockBackend(response=mock_response)
        agent = LLMAnomalyAgent(backend=backend, batch_size=5)

        result = agent.analyze_single(
            window_values=np.random.randn(50),
            evidence={'mae': 5.0, 'max_abs_z_score': 4.0},
            window_index=0
        )

        assert len(result["windows"]) == 1
        assert result["windows"][0]["confidence"] == 0.9
        assert agent.get_call_count() == 1
        assert len(backend.calls) == 1

    def test_agent_batch_analysis(self):
        from src.llm.llm_agent import LLMAnomalyAgent

        mock_response = json.dumps({
            "windows": [
                {"window_index": 0, "confidence": 0.1, "is_anomaly": False},
                {"window_index": 1, "confidence": 0.9, "is_anomaly": True},
                {"window_index": 2, "confidence": 0.3, "is_anomaly": False},
            ]
        })
        backend = MockBackend(response=mock_response)
        agent = LLMAnomalyAgent(backend=backend, batch_size=10)

        windows = np.random.randn(3, 50, 1)
        evidence_list = [
            {'mae': 0.5, 'max_abs_z_score': 1.0},
            {'mae': 5.0, 'max_abs_z_score': 4.5},
            {'mae': 0.8, 'max_abs_z_score': 1.2},
        ]

        scores = agent.analyze_windows(windows, evidence_list, progress=False)

        assert scores.shape == (3,)
        assert scores[1] == 0.9  # Anomalous window
        assert scores[0] == 0.1  # Normal window
        assert agent.get_call_count() == 1  # Single batch


# ============================================================
# Backend Factory Test
# ============================================================

class TestBackendFactory:

    def test_create_backend_invalid(self):
        from src.llm.backends import create_backend
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("invalid_backend")

    def test_create_backend_types(self):
        from src.llm.backends import create_backend
        # These will fail without API keys, but should raise import/config errors, not ValueError
        for backend_type in ["azure_openai", "gemini", "claude"]:
            try:
                create_backend(backend_type)
            except (ValueError, Exception):
                pass  # Expected without API keys


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
