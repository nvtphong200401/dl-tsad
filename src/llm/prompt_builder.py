"""Prompt Builder for LLM Anomaly Detection

Formats statistical evidence and time series data into structured
prompts for LLM reasoning.
"""

import numpy as np
from typing import Dict, List, Optional


SYSTEM_PROMPT = """You are an expert time series analyst performing anomaly detection.

CRITICAL CONTEXT:
- These candidate windows are the top 10% most statistically suspicious from ~90 windows.
- Being in the top 10% does NOT mean anomalous. In most time series, 0-3% of windows contain real anomalies.
- Expect 0-1 truly anomalous windows per batch. Many batches will have ZERO anomalies.
- The baseline windows show what NORMAL looks like. Candidates with similar metric magnitudes are NORMAL.

IMPORTANT - common false positive traps:
- High KL divergence alone is NOT an anomaly indicator (different sampling regions cause this naturally)
- Violation ratio close to 1.0 can be normal if baseline windows also show high violation ratios
- Z-scores of 3-5 can occur naturally in the tails of normal distributions
- Only flag anomalies when metrics are DRAMATICALLY higher than baseline (e.g., 3-10x)

Decision criteria (ALL must be met):
1. Multiple metrics must be elevated SIMULTANEOUSLY (not just one)
2. The values must be dramatically different from baseline windows (not just slightly higher)
3. The raw time series values should show visible deviation (sudden spike, level shift, or pattern break)

Confidence calibration:
- 0.0-0.3: Normal (default for most windows)
- 0.3-0.5: Slightly unusual, probably noise
- 0.5-0.7: Suspicious but not conclusive
- 0.7-0.85: Likely anomaly (metrics 5x+ above baseline with convergent evidence)
- 0.85-1.0: Clear anomaly (extreme deviation across all metrics)

Output valid JSON with ONLY the candidate windows (not baseline windows):
{
  "windows": [
    {
      "window_index": 0,
      "is_anomaly": false,
      "confidence": 0.1,
      "reasoning": "Metrics within normal range for this series.",
      "evidence_cited": []
    }
  ]
}

Rules:
- confidence between 0.0 and 1.0
- is_anomaly = true ONLY if confidence >= 0.7
- Expect MOST or ALL candidates to be normal (confidence < 0.3)
- It is completely valid to flag ZERO anomalies in a batch
- Keep reasoning concise (1 sentence)
- Do NOT include baseline windows in output"""


def _compute_baseline_averages(baseline_evidence: List[Dict]) -> Dict[str, float]:
    """Compute average metric values from baseline evidence for normalization."""
    if not baseline_evidence:
        return {}
    keys = ['mae', 'mse', 'mape', 'max_abs_z_score', 'grubbs_statistic',
            'max_cusum', 'kl_divergence', 'normalized_wasserstein',
            'volatility_ratio', 'violation_ratio', 'mean_surprise',
            'max_acf_diff', 'slope_diff']
    avgs = {}
    for k in keys:
        vals = [e.get(k) for e in baseline_evidence if isinstance(e.get(k), (int, float))]
        if vals:
            avgs[k] = max(sum(vals) / len(vals), 1e-10)
    return avgs


def _format_relative_evidence(evidence: Dict, baseline_avgs: Dict[str, float]) -> str:
    """Format evidence as ratios relative to baseline averages."""
    lines = []
    metrics = [
        ('mae', 'MAE'),
        ('mse', 'MSE'),
        ('max_abs_z_score', 'Max Z-Score'),
        ('volatility_ratio', 'Volatility Ratio'),
        ('kl_divergence', 'KL Divergence'),
        ('violation_ratio', 'Quantile Violation Rate'),
        ('mean_surprise', 'Surprise Score'),
        ('max_cusum', 'CUSUM Max'),
        ('grubbs_statistic', 'Grubbs Statistic'),
        ('normalized_wasserstein', 'Wasserstein Distance'),
        ('max_acf_diff', 'ACF Break'),
        ('slope_diff', 'Slope Diff'),
    ]
    for key, label in metrics:
        if key not in evidence or not isinstance(evidence[key], (int, float)):
            continue
        val = evidence[key]
        if key in baseline_avgs and baseline_avgs[key] > 1e-10:
            ratio = val / baseline_avgs[key]
            if ratio > 3.0:
                flag = " **ELEVATED**"
            elif ratio > 1.5:
                flag = " (slightly above)"
            else:
                flag = ""
            lines.append(f"- {label}: {val:.3f} ({ratio:.1f}x baseline){flag}")
        else:
            lines.append(f"- {label}: {val:.3f}")

    # Boolean/special fields
    if evidence.get('grubbs_is_outlier'):
        lines.append(f"- Grubbs Test: OUTLIER at position {evidence.get('grubbs_outlier_index', '?')}")
    if evidence.get('cusum_has_change_point'):
        lines.append(f"- CUSUM: Change point detected")
    if evidence.get('trend_break'):
        lines.append(f"- Trend Break: YES")
    if evidence.get('period_changed'):
        lines.append(f"- Period Changed: YES")

    return "\n".join(lines) if lines else "No evidence available"


def build_batch_prompt(
    windows: List[np.ndarray],
    evidence_list: List[Dict],
    window_indices: List[int],
    baseline_windows: Optional[List[np.ndarray]] = None,
    baseline_evidence: Optional[List[Dict]] = None
) -> str:
    """Build prompt for a batch of windows with baseline-relative evidence.

    Args:
        windows: List of window arrays (W,) or (W, D)
        evidence_list: List of evidence dicts from Step 2
        window_indices: Original window indices
        baseline_windows: Optional normal windows for contrast
        baseline_evidence: Optional evidence for normal windows

    Returns:
        Formatted user prompt string
    """
    sections = ["# Time Series Anomaly Analysis\n"]

    # Compute baseline averages for relative scoring
    baseline_avgs = _compute_baseline_averages(baseline_evidence) if baseline_evidence else {}

    if baseline_avgs:
        sections.append("## Baseline Reference (average of confirmed normal windows)\n")
        for key, label in [('mae', 'MAE'), ('max_abs_z_score', 'Z-Score'),
                           ('kl_divergence', 'KL Div'), ('volatility_ratio', 'Volatility'),
                           ('violation_ratio', 'Violation Rate')]:
            if key in baseline_avgs:
                sections.append(f"- {label}: {baseline_avgs[key]:.3f}")
        sections.append("\nCandidate metrics are shown as ratios vs these baselines.")
        sections.append("Only flag windows where metrics are **3x+ above baseline** across multiple metrics.\n")

    # Show candidate windows with relative evidence
    sections.append(f"## Candidate Windows ({len(windows)} windows)\n")

    for window, evidence, idx in zip(windows, evidence_list, window_indices):
        sections.append(f"---\n### Window {idx}\n")

        values = window.squeeze() if hasattr(window, 'squeeze') else window
        sections.append(format_time_series(values))

        sections.append("\n**Evidence (vs baseline):**")
        sections.append(_format_relative_evidence(evidence, baseline_avgs))
        sections.append("")

    sections.append(f"\n---\nAnalyze the {len(windows)} windows. Return JSON. Most should be confidence < 0.3.")
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
