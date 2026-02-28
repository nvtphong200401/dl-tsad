"""Prompt Builder for AnoAgent-style LLM Range Detection

Formats full time series with derived signals (CUSUM, running mean) and
asks LLM to return anomaly ranges directly. Adapted from the AnoAgent paper.
"""

import numpy as np
from typing import Optional, Tuple


RANGE_DETECTION_SYSTEM_PROMPT = """You are an expert time series anomaly detector. Be extremely conservative.

You will receive a time series with 3 channels per data point:
- R (Residual): The deseasonalized signal. Spikes/dips indicate point anomalies.
- C (CUSUM): Cumulative deviation detector. Ramps up at the ONSET of a mean shift. Note: C may reset after peaking — it detects the transition, not the full extent.
- M (Running Mean): Local mean vs global mean. Near 5.0 when normal. **M is the best indicator for the FULL EXTENT of an anomaly** — it stays elevated/depressed throughout the entire anomalous region.

HOW TO DETECT ANOMALY RANGES:
1. Use C to find where an anomaly STARTS (where C first rises above 0)
2. Use M to find where the anomaly ENDS (where M returns to ~5.0)
3. The anomaly range = from where C first rises TO where M returns to normal
4. Point anomaly: R shows a clear spike at a single location

CRITICAL: Most time series are NORMAL (0 anomalies). M staying near 5.0 everywhere = no anomaly.

RULES:
- Return anomaly ranges as JSON: {{"anomalies": [{{"start": <int>, "end": <int>}}]}}
- "start" is inclusive, "end" is exclusive
- Return at most {max_anomalies} anomaly ranges, but prefer FEWER
- If no clear anomalies exist, return {{"anomalies": []}}
- The anomaly range should cover the FULL region where M deviates from 5.0
- When in doubt, do NOT flag

Output ONLY valid JSON, no explanations."""


def _scale_to_range(arr: np.ndarray, lo: float = 0.0, hi: float = 10.0) -> np.ndarray:
    """Min-max scale array to [lo, hi]."""
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min > 1e-10:
        return (arr - a_min) / (a_max - a_min) * (hi - lo) + lo
    return np.full_like(arr, (lo + hi) / 2)


def compute_derived_signals(
    series: np.ndarray,
    cusum_slack_factor: float = 3.0,
    running_mean_window: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CUSUM and running mean deviation from a 1D series.

    Args:
        series: 1D time series (T,)
        cusum_slack_factor: CUSUM slack as fraction of std (default 0.5)
        running_mean_window: Window size for running mean (default 50)

    Returns:
        (cusum, running_mean_dev) - both 1D arrays of length T
    """
    T = len(series)
    mean = float(np.mean(series))
    std = float(np.std(series))
    slack = cusum_slack_factor * max(std, 1e-10)

    # Bidirectional CUSUM
    cusum_pos = np.zeros(T)
    cusum_neg = np.zeros(T)
    for i in range(1, T):
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + (series[i] - mean) - slack)
        cusum_neg[i] = max(0.0, cusum_neg[i - 1] - (series[i] - mean) - slack)
    cusum = np.maximum(cusum_pos, cusum_neg)

    # Running mean deviation
    w = min(running_mean_window, T)
    cumsum = np.cumsum(series)
    running_mean = np.zeros(T)
    for i in range(T):
        start = max(0, i - w + 1)
        running_mean[i] = (cumsum[i] - (cumsum[start - 1] if start > 0 else 0)) / (i - start + 1)
    running_mean_dev = running_mean - mean

    return cusum, running_mean_dev


def build_range_detection_prompt(
    series: np.ndarray,
    max_anomalies: int,
    evidence_summary: Optional[str] = None,
    scale_range: tuple = (0, 10),
    max_digits: int = 1,
) -> str:
    """Build multi-channel prompt for full-series range detection.

    Includes 3 channels: Residual, CUSUM, Running Mean.

    Args:
        series: 1D time series (T,) - deseasonalized residual
        max_anomalies: Maximum number of anomaly ranges to return
        evidence_summary: Optional text summary of suspicious regions
        scale_range: Target range for normalization (default [0, 10])
        max_digits: Decimal precision for values

    Returns:
        Formatted user prompt string
    """
    # Compute derived signals
    cusum, running_mean_dev = compute_derived_signals(series)

    # Scale residual to [0, 10] via min-max
    r_scaled = _scale_to_range(series, *scale_range)

    # Scale CUSUM relative to residual std (not min-max).
    # Normal CUSUM stays near 0; anomalous CUSUM reaches 5-10.
    std = max(float(np.std(series)), 1e-10)
    cusum_ref = 10.0 * std  # CUSUM above 10*std maps to 10
    c_scaled = np.clip(cusum / max(cusum_ref, 1e-10) * 10.0, 0.0, 10.0)

    # Scale running mean: center at 5.0, deviation of 1 std = ±2.5
    m_scaled = np.clip(5.0 + running_mean_dev / max(std, 1e-10) * 2.5, 0.0, 10.0)

    # Format as (index, R, C, M) tuples — compact multi-channel format
    d = max_digits
    history = "\n".join(
        f"({i}, {r_scaled[i]:.{d}f}, {c_scaled[i]:.{d}f}, {m_scaled[i]:.{d}f})"
        for i in range(len(series))
    )

    sections = [
        "Time series with 3 channels: (index, R=Residual, C=CUSUM, M=RunningMean).",
        "All channels scaled to [0, 10].",
        "",
        "<history>",
        history,
        "</history>",
        "",
        f"Series length: {len(series)} points.",
        f"There may be 0 to {max_anomalies} anomalies. Most series have 0-2.",
    ]

    if evidence_summary:
        sections.extend([
            "",
            "## Statistical Hints:",
            evidence_summary,
        ])

    sections.extend([
        "",
        "Detect anomaly ranges. Look for regions where C rises sharply from baseline.",
        f"Return at most {max_anomalies} ranges.",
        'If no anomalies, return {"anomalies": []}.',
        "",
        '{"anomalies": [{"start": ..., "end": ...}]}',
    ])

    return "\n".join(sections)


def build_evidence_summary(
    subsequence_scores: np.ndarray,
    window_size: int,
    stride: int,
    top_k: int = 5,
) -> str:
    """Summarize evidence scores to hint at suspicious regions."""
    N = len(subsequence_scores)
    top_k = min(top_k, N)
    top_indices = np.argsort(subsequence_scores)[-top_k:][::-1]

    lines = [f"Top {top_k} suspicious regions:"]
    for idx in top_indices:
        start = idx * stride
        end = start + window_size
        score = subsequence_scores[idx]
        lines.append(f"- indices {start}-{end}: score {score:.3f}")

    mean_score = float(np.mean(subsequence_scores))
    p95_score = float(np.percentile(subsequence_scores, 95))
    lines.append(f"Mean score: {mean_score:.3f}, P95: {p95_score:.3f}")

    return "\n".join(lines)
