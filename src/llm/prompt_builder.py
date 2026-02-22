"""Prompt Builder for LLM Anomaly Detection

Formats statistical evidence and time series data into structured
prompts for LLM reasoning.
"""

import numpy as np
from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert time series analyst. Your task is to identify ONLY truly anomalous windows from statistical evidence.

CRITICAL: Most windows are NORMAL. Typically only 3-10% of windows contain real anomalies. Be conservative — only flag a window as anomalous when you see STRONG, CONVERGENT evidence from MULTIPLE metrics.

Decision thresholds (a single metric alone is NOT enough):
- Z-score > 3.0 AND at least one other signal → possible anomaly
- Volatility ratio > 3.0x AND distribution shift → possible anomaly
- CUSUM change point AND trend break → possible anomaly
- Moderate values across all metrics → NORMAL (this is baseline noise)

Confidence calibration:
- 0.0-0.2: Normal window, no concerning signals
- 0.2-0.4: Slightly unusual but likely noise
- 0.4-0.6: Ambiguous, some signals but not conclusive
- 0.6-0.8: Likely anomaly, multiple convergent signals
- 0.8-1.0: Clear anomaly, strong convergent evidence

Output valid JSON:
{
  "windows": [
    {
      "window_index": 0,
      "is_anomaly": false,
      "confidence": 0.15,
      "reasoning": "All metrics within normal range.",
      "evidence_cited": []
    }
  ]
}

Rules:
- confidence between 0.0 and 1.0
- is_anomaly = true ONLY if confidence >= 0.6
- Most windows should have confidence < 0.3
- Keep reasoning concise (1 sentence)"""


def build_batch_prompt(
    windows: List[np.ndarray],
    evidence_list: List[Dict],
    window_indices: List[int]
) -> str:
    """Build prompt for a batch of windows.

    Args:
        windows: List of window arrays (W,) or (W, D)
        evidence_list: List of evidence dicts from Step 2
        window_indices: Original window indices

    Returns:
        Formatted user prompt string
    """
    sections = ["# Time Series Anomaly Analysis\n"]
    sections.append(f"Analyze the following {len(windows)} windows and determine if each contains anomalies.\n")

    for i, (window, evidence, idx) in enumerate(zip(windows, evidence_list, window_indices)):
        sections.append(f"---\n## Window {idx}\n")

        # Time series data
        values = window.squeeze() if hasattr(window, 'squeeze') else window
        sections.append(format_time_series(values))

        # Evidence
        sections.append("\n### Statistical Evidence\n")
        sections.append(format_forecast_evidence(evidence))
        sections.append(format_statistical_tests(evidence))
        sections.append(format_distribution_evidence(evidence))
        sections.append(format_pattern_evidence(evidence))

    sections.append("\n---\nPlease analyze all windows above and return JSON output.")
    return "\n".join(sections)


def build_single_prompt(
    window_values: np.ndarray,
    evidence: Dict,
    window_index: int
) -> str:
    """Build prompt for a single window.

    Args:
        window_values: Window data (W,)
        evidence: Evidence dict from Step 2
        window_index: Window index

    Returns:
        Formatted user prompt string
    """
    return build_batch_prompt([window_values], [evidence], [window_index])


def format_time_series(values: np.ndarray, max_points: int = 50) -> str:
    """Format time series values for prompt.

    Shows summary stats + sampled points for long series.
    """
    values = np.asarray(values).flatten()
    n = len(values)

    lines = [f"**Data** ({n} points): mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
             f"min={np.min(values):.4f}, max={np.max(values):.4f}"]

    if n <= max_points:
        vals = ", ".join(f"{v:.3f}" for v in values)
        lines.append(f"Values: [{vals}]")
    else:
        # Show first 10, last 10
        first = ", ".join(f"{v:.3f}" for v in values[:10])
        last = ", ".join(f"{v:.3f}" for v in values[-10:])
        lines.append(f"First 10: [{first}]")
        lines.append(f"Last 10: [{last}]")

    return "\n".join(lines)


def format_forecast_evidence(evidence: Dict) -> str:
    """Format forecast-based evidence metrics."""
    lines = []

    if 'mae' in evidence:
        lines.append(f"- MAE (Mean Absolute Error): {evidence['mae']:.4f}")

    if 'mse' in evidence:
        lines.append(f"- MSE (Mean Squared Error): {evidence['mse']:.4f}")

    if 'mape' in evidence:
        lines.append(f"- MAPE: {evidence['mape']:.1f}%")

    if 'violation_ratio' in evidence:
        ratio = evidence['violation_ratio']
        status = "ANOMALOUS" if ratio > 0.1 else "normal"
        lines.append(f"- Quantile Violation Ratio: {ratio:.2f} ({status})")

    if 'extreme_violation' in evidence:
        if evidence['extreme_violation']:
            lines.append("- Extreme Quantile Violation: YES (value outside P01/P99)")

    if 'mean_surprise' in evidence:
        surprise = evidence['mean_surprise']
        status = "HIGH" if surprise > 2.0 else "normal"
        lines.append(f"- Surprise Score: {surprise:.2f} ({status})")

    if not lines:
        return ""
    return "**Forecast-Based Evidence:**\n" + "\n".join(lines) + "\n"


def format_statistical_tests(evidence: Dict) -> str:
    """Format statistical test evidence metrics."""
    lines = []

    if 'max_abs_z_score' in evidence:
        z = evidence['max_abs_z_score']
        status = "ANOMALOUS (>3-sigma)" if z > 3 else "normal"
        lines.append(f"- Max Z-Score: {z:.2f} ({status})")

    if 'extreme_z_count' in evidence:
        lines.append(f"- Points exceeding 3-sigma: {evidence['extreme_z_count']}")

    if 'grubbs_is_outlier' in evidence:
        if evidence['grubbs_is_outlier']:
            idx = evidence.get('grubbs_outlier_index', '?')
            lines.append(f"- Grubbs Test: OUTLIER detected at position {idx}")
        else:
            lines.append("- Grubbs Test: No outlier")

    if 'max_cusum' in evidence:
        cusum = evidence['max_cusum']
        has_cp = evidence.get('cusum_has_change_point', False)
        if has_cp:
            lines.append(f"- CUSUM: Change point detected (max={cusum:.2f})")
        else:
            lines.append(f"- CUSUM: No change point (max={cusum:.2f})")

    if not lines:
        return ""
    return "**Statistical Tests:**\n" + "\n".join(lines) + "\n"


def format_distribution_evidence(evidence: Dict) -> str:
    """Format distribution-based evidence metrics."""
    lines = []

    if 'kl_divergence' in evidence:
        kl = evidence['kl_divergence']
        status = "HIGH divergence" if kl > 0.5 else "low"
        lines.append(f"- KL Divergence: {kl:.4f} ({status})")

    if 'normalized_wasserstein' in evidence:
        w = evidence['normalized_wasserstein']
        status = "SIGNIFICANT shift" if w > 1.0 else "minor"
        lines.append(f"- Wasserstein Distance (normalized): {w:.4f} ({status})")

    if not lines:
        return ""
    return "**Distribution Analysis:**\n" + "\n".join(lines) + "\n"


def format_pattern_evidence(evidence: Dict) -> str:
    """Format pattern-based evidence metrics."""
    lines = []

    if 'volatility_ratio' in evidence:
        vr = evidence['volatility_ratio']
        if vr > 5.0:
            status = "EXTREME (>5x baseline)"
        elif vr > 2.0:
            status = "HIGH (>2x baseline)"
        else:
            status = "normal"
        lines.append(f"- Volatility Ratio: {vr:.2f}x ({status})")

    if 'max_acf_diff' in evidence:
        acf = evidence['max_acf_diff']
        lines.append(f"- ACF Break: max diff={acf:.4f}")

    if 'period_changed' in evidence and evidence['period_changed']:
        lines.append("- Periodicity: CHANGED from training pattern")

    if 'slope_diff' in evidence:
        sd = evidence['slope_diff']
        if evidence.get('trend_break', False):
            lines.append(f"- Trend Break: YES (slope diff={sd:.4f})")
        else:
            lines.append(f"- Trend: stable (slope diff={sd:.4f})")

    if not lines:
        return ""
    return "**Pattern Analysis:**\n" + "\n".join(lines) + "\n"
