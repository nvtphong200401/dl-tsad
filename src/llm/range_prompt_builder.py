"""Prompt Builder for AnoAgent-style LLM Range Detection

Formats full time series as (index, value) pairs and asks LLM
to return anomaly ranges directly. Adapted from the AnoAgent paper.
"""

import numpy as np
from typing import Optional


RANGE_DETECTION_SYSTEM_PROMPT = """You are an expert time series anomaly detector.

You will receive a deseasonalized time series as (index, value) pairs.
The seasonal component has already been removed, so focus on:
- Sudden spikes or dips
- Level shifts
- Unusual volatility changes
- Values that deviate significantly from the local trend

RULES:
- Return anomaly ranges as JSON: {{"anomalies": [{{"start": <int>, "end": <int>}}]}}
- "start" is inclusive, "end" is exclusive
- Each range should be tight (cover only the actual anomalous points)
- Return at most {max_anomalies} anomaly ranges
- If no anomalies exist, return {{"anomalies": []}}
- Keep ranges short. Most anomalies span 5-50 points, rarely more than 100
- Do NOT flag normal fluctuations or noise as anomalies
- Be conservative: only flag clear anomalies, not borderline cases

Output ONLY valid JSON, no explanations."""


def build_range_detection_prompt(
    series: np.ndarray,
    max_anomalies: int,
    evidence_summary: Optional[str] = None,
    scale_range: tuple = (0, 10),
    max_digits: int = 2,
) -> str:
    """Build AnoAgent-style prompt for full-series range detection.

    Args:
        series: 1D time series (T,) - deseasonalized
        max_anomalies: Maximum number of anomaly ranges to return
        evidence_summary: Optional text summary of suspicious regions
        scale_range: Target range for normalization (default [0, 10])
        max_digits: Decimal precision for values

    Returns:
        Formatted user prompt string
    """
    # Min-max scale to [0, 10]
    s_min, s_max = float(series.min()), float(series.max())
    if s_max - s_min > 1e-10:
        scaled = (series - s_min) / (s_max - s_min) * (scale_range[1] - scale_range[0]) + scale_range[0]
    else:
        scaled = np.full_like(series, (scale_range[0] + scale_range[1]) / 2)

    # Format as (index, value) pairs
    history = "\n".join(
        f"({i}, {v:.{max_digits}f})"
        for i, v in enumerate(scaled)
    )

    sections = [
        "I will provide you with a deseasonalized time-series (seasonal pattern removed).",
        "",
        f"Here is time-series data in (index, value) format:",
        f"<history>",
        history,
        f"</history>",
        "",
        f"Assume there are at most {max_anomalies} anomalies.",
        f"The index of the time series starts from 0 to {len(series) - 1}.",
    ]

    if evidence_summary:
        sections.extend([
            "",
            "## Statistical Hints (from automated pre-analysis):",
            evidence_summary,
        ])

    sections.extend([
        "",
        "Detect ranges of anomalies in this time series.",
        f"Return at most {max_anomalies} ranges. Most series have 0-3 anomalies.",
        "List one by one, in JSON format.",
        'If there are no anomalies, return {"anomalies": []}.',
        "Do not say anything other than the JSON answer.",
        "",
        'Output template:',
        '{"anomalies": [{"start": ..., "end": ...}, {"start": ..., "end": ...}]}',
    ])

    return "\n".join(sections)


def build_evidence_summary(
    subsequence_scores: np.ndarray,
    window_size: int,
    stride: int,
    top_k: int = 5,
) -> str:
    """Summarize evidence scores to hint at suspicious regions.

    Args:
        subsequence_scores: Window-level scores from Step 2 (N,)
        window_size: Window size
        stride: Stride
        top_k: Number of top windows to mention

    Returns:
        Text summary of suspicious regions
    """
    N = len(subsequence_scores)
    top_k = min(top_k, N)
    top_indices = np.argsort(subsequence_scores)[-top_k:][::-1]

    lines = [f"The top {top_k} most statistically suspicious regions:"]
    for idx in top_indices:
        start = idx * stride
        end = start + window_size
        score = subsequence_scores[idx]
        lines.append(f"- Region indices {start}-{end}: anomaly score {score:.3f}")

    mean_score = float(np.mean(subsequence_scores))
    p95_score = float(np.percentile(subsequence_scores, 95))
    lines.append("")
    lines.append(f"Average window score: {mean_score:.3f}, 95th percentile: {p95_score:.3f}")

    return "\n".join(lines)
