"""Output Parser for LLM Range Detection Responses

Parses [{"start": ..., "end": ...}] ranges from LLM and converts
to point-level binary scores.
"""

import json
import re
import numpy as np
from typing import Dict, List


def parse_range_output(raw_text: str) -> Dict:
    """Parse LLM response containing anomaly ranges.

    Handles:
    - Raw JSON with "anomalies" key
    - JSON in markdown code blocks
    - Direct array of {"start", "end"} dicts

    Args:
        raw_text: Raw text response from LLM

    Returns:
        Dict with 'anomalies' key: list of {"start": int, "end": int}
        Includes 'parse_error' key if parsing failed.
    """
    if not raw_text or not raw_text.strip():
        return {"anomalies": [], "parse_error": "Empty LLM response"}

    for candidate in _extract_json_candidates(raw_text):
        try:
            parsed = json.loads(candidate)
            result = _validate_ranges(parsed)
            if result.get("anomalies") is not None:
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    return {"anomalies": [], "parse_error": f"Could not parse: {raw_text[:200]}"}


def _extract_json_candidates(raw_text: str) -> List[str]:
    """Extract possible JSON strings from raw text."""
    candidates = [raw_text.strip()]

    # JSON in code blocks
    for match in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?```', raw_text, re.DOTALL):
        candidates.append(match.group(1).strip())

    # Brace-delimited JSON object
    for match in re.finditer(r'\{.*\}', raw_text, re.DOTALL):
        candidates.append(match.group(0))

    # Bracket-delimited array
    for match in re.finditer(r'\[.*\]', raw_text, re.DOTALL):
        candidates.append(match.group(0))

    return candidates


def _validate_ranges(parsed) -> Dict:
    """Validate and normalize parsed range output."""
    # Handle dict with "anomalies" key
    if isinstance(parsed, dict):
        anomalies = parsed.get("anomalies", [])
    elif isinstance(parsed, list):
        anomalies = parsed
    else:
        return {"anomalies": [], "parse_error": "Unexpected format"}

    validated = []
    for a in anomalies:
        if isinstance(a, dict) and "start" in a and "end" in a:
            start = int(a["start"])
            end = int(a["end"])
            # Handle both inclusive and exclusive end conventions
            if start == end:
                end = start + 1  # Single point
            if start < end:
                validated.append({"start": start, "end": end})

    return {"anomalies": validated, "parse_error": None}


def ranges_to_point_scores(
    anomaly_ranges: List[Dict],
    series_length: int,
) -> np.ndarray:
    """Convert anomaly ranges to point-level scores.

    Points inside any anomaly range get score 1.0.
    Points outside all ranges get score 0.0.

    Args:
        anomaly_ranges: List of {"start": int, "end": int} dicts
        series_length: Length of original series

    Returns:
        Point-level scores (T,) with values 0.0 or 1.0
    """
    scores = np.zeros(series_length)

    for r in anomaly_ranges:
        start = max(0, r["start"])
        end = min(series_length, r["end"])
        if start < end:
            scores[start:end] = 1.0

    return scores
