"""Output Parser for LLM Anomaly Detection Responses

Parses JSON from LLM responses with fallback handling.
"""

import json
import re
from typing import Dict, List, Optional


def parse_llm_output(raw_text: str) -> Dict:
    """Parse LLM response into structured result.

    Handles:
    - Raw JSON
    - JSON in markdown code blocks
    - Partial/malformed JSON with fallback

    Args:
        raw_text: Raw text response from LLM

    Returns:
        Parsed dict with 'windows' key containing analysis per window.
        Includes 'parse_error' key if parsing failed.
    """
    if not raw_text or not raw_text.strip():
        return _default_result("Empty LLM response")

    # Try direct JSON parse
    try:
        parsed = json.loads(raw_text.strip())
        return validate_result(parsed)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            return validate_result(parsed)
        except json.JSONDecodeError:
            pass

    # Try finding JSON object in text
    brace_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            return validate_result(parsed)
        except json.JSONDecodeError:
            pass

    # All parsing failed
    return _default_result(f"Could not parse JSON from response: {raw_text[:200]}")


def validate_result(parsed: Dict) -> Dict:
    """Validate and normalize parsed LLM output.

    Ensures required fields exist and values are in valid ranges.
    """
    result = {
        "windows": [],
        "parse_error": None,
        "raw_parsed": parsed
    }

    # Handle 'windows' key (batch format)
    windows = parsed.get("windows", [])

    # Handle 'anomalies' key (single-window format from spec)
    if not windows and "anomalies" in parsed:
        # Convert spec format to our batch format
        anomalies = parsed["anomalies"]
        is_anomaly = len(anomalies) > 0
        confidence = max((a.get("confidence", 0.5) for a in anomalies), default=0.0)
        reasoning = parsed.get("overall_assessment", "")
        evidence = []
        for a in anomalies:
            evidence.extend(a.get("evidence_cited", []))

        windows = [{
            "window_index": 0,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence_cited": evidence
        }]

    # Validate each window result
    validated_windows = []
    for w in windows:
        validated = {
            "window_index": w.get("window_index", 0),
            "is_anomaly": bool(w.get("is_anomaly", False)),
            "confidence": _clamp(float(w.get("confidence", 0.0)), 0.0, 1.0),
            "reasoning": str(w.get("reasoning", "")),
            "evidence_cited": list(w.get("evidence_cited", []))
        }
        validated_windows.append(validated)

    result["windows"] = validated_windows
    return result


def extract_window_confidence(
    parsed_result: Dict,
    window_indices: List[int]
) -> Dict[int, float]:
    """Extract confidence scores mapped to window indices.

    Args:
        parsed_result: Output from parse_llm_output()
        window_indices: List of window indices that were analyzed

    Returns:
        Dict mapping window_index -> confidence score
    """
    confidence_map = {}

    for w in parsed_result.get("windows", []):
        idx = w.get("window_index", -1)
        confidence_map[idx] = w.get("confidence", 0.0)

    # Fill in any missing windows with 0.0
    for idx in window_indices:
        if idx not in confidence_map:
            confidence_map[idx] = 0.0

    return confidence_map


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _default_result(error_msg: str) -> Dict:
    """Return default result when parsing fails."""
    return {
        "windows": [],
        "parse_error": error_msg,
        "raw_parsed": None
    }
