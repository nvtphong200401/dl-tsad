"""Step 3: Scoring - Convert sub-sequence scores to point-wise scores"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List
import numpy as np


class ScoringMethod(ABC):
    """Base class for Step 3: Convert sub-sequence scores to point-wise scores

    Scoring methods aggregate window-level anomaly scores to point-level scores.
    """

    @abstractmethod
    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Convert sub-sequence scores to point-wise scores

        Args:
            subsequence_scores: Window-level scores (N,)
            window_size: Size of each window
            stride: Stride between windows
            original_length: Length of original time series

        Returns:
            Point-wise scores (T,)
        """
        pass


class MaxPoolingScoring(ScoringMethod):
    """Each point gets maximum score from all windows containing it

    This is conservative - emphasizes detecting anomalies (high recall).
    """

    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Max pooling aggregation"""
        point_scores = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start = i * stride
            end = min(start + window_size, original_length)
            # Max pooling - take maximum
            point_scores[start:end] = np.maximum(point_scores[start:end], score)

        return point_scores


class AveragePoolingScoring(ScoringMethod):
    """Each point gets average score from all windows containing it

    This is balanced - averages contributions from overlapping windows.
    """

    def score(self,
              subsequence_scores: np.ndarray,
              window_size: int,
              stride: int,
              original_length: int) -> np.ndarray:
        """Average pooling aggregation"""
        point_scores = np.zeros(original_length)
        counts = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start = i * stride
            end = min(start + window_size, original_length)
            # Sum scores
            point_scores[start:end] += score
            counts[start:end] += 1

        # Average (avoid division by zero)
        point_scores = point_scores / np.maximum(counts, 1)

        return point_scores


class LLMReasoningScoring(ScoringMethod):
    """LLM-based scoring with statistical pre-filtering.

    Two-stage approach:
    1. Pre-filter: Use statistical scores from Step 2 to identify the
       most suspicious windows (top K%).
    2. LLM reasoning: Only send suspicious windows to the LLM for
       detailed analysis. Normal windows keep their statistical score.

    This reduces API calls by 80-90% while focusing LLM reasoning
    where it matters most — on borderline and high-score windows.

    The orchestrator injects evidence via set_evidence_context() before
    calling score().
    """

    def __init__(
        self,
        backend_type: str = "azure_openai",
        batch_size: int = 10,
        temperature: Optional[float] = None,
        pre_filter_percentile: float = 80.0,
        **backend_kwargs
    ):
        """Initialize LLM scoring.

        Args:
            backend_type: "azure_openai", "gemini", or "claude"
            batch_size: Windows per LLM call (default: 10)
            temperature: LLM temperature
            pre_filter_percentile: Only send windows scoring above this
                percentile to the LLM. 80.0 = top 20% of windows.
                Set to 0.0 to disable pre-filtering (send all windows).
            **backend_kwargs: Passed to backend constructor
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from llm import create_backend, LLMAnomalyAgent

        self.backend = create_backend(backend_type, **backend_kwargs)
        self.agent = LLMAnomalyAgent(
            backend=self.backend,
            batch_size=batch_size,
            temperature=temperature
        )
        self.pre_filter_percentile = pre_filter_percentile

        # State set via set_evidence_context()
        self.evidence_list = None
        self.windows = None

        # Results
        self.llm_scores = None

    def set_evidence_context(
        self,
        evidence_list: List[Dict],
        windows: np.ndarray
    ) -> None:
        """Inject evidence from Step 2.

        Called by the orchestrator between Step 2 and Step 3.

        Args:
            evidence_list: List of evidence dicts from EvidenceBasedDetection
            windows: Processed windows (N, W, D) from Step 1
        """
        self.evidence_list = evidence_list
        self.windows = windows

    def score(
        self,
        subsequence_scores: np.ndarray,
        window_size: int,
        stride: int,
        original_length: int
    ) -> np.ndarray:
        """Score using pre-filtered LLM reasoning.

        Stage 1: Identify suspicious windows via statistical pre-filter.
        Stage 2: Send only suspicious windows to LLM for reasoning.
        Non-suspicious windows keep their normalized statistical score.

        Args:
            subsequence_scores: Window-level scores from Step 2 (N,)
            window_size: Size of each window
            stride: Stride between windows
            original_length: Length of original time series

        Returns:
            Point-wise scores (T,)
        """
        N = len(subsequence_scores)
        scores_to_pool = subsequence_scores.copy()

        if self.evidence_list is not None and self.windows is not None:
            # Stage 1: Pre-filter — identify suspicious windows
            if self.pre_filter_percentile > 0 and N > 0:
                threshold = np.percentile(subsequence_scores, self.pre_filter_percentile)
                suspicious_mask = subsequence_scores >= threshold
                # Always send at least 1 window
                if not np.any(suspicious_mask):
                    suspicious_mask[np.argmax(subsequence_scores)] = True
            else:
                suspicious_mask = np.ones(N, dtype=bool)

            suspicious_indices = np.where(suspicious_mask)[0]
            n_suspicious = len(suspicious_indices)
            n_skipped = N - n_suspicious

            print(f"  Pre-filter: {n_suspicious}/{N} windows sent to LLM "
                  f"({n_skipped} skipped, threshold=P{self.pre_filter_percentile:.0f})")

            # Select normal contrast windows (lowest-scoring, clearly normal)
            normal_mask = ~suspicious_mask
            normal_indices = np.where(normal_mask)[0]
            n_contrast = min(3, len(normal_indices))
            if n_contrast > 0:
                # Pick windows with lowest scores as contrast
                lowest_indices = normal_indices[
                    np.argsort(subsequence_scores[normal_indices])[:n_contrast]
                ]
                baseline_wins = [
                    self.windows[i, :, 0] if self.windows.ndim == 3
                    else self.windows[i]
                    for i in lowest_indices
                ]
                baseline_evid = [self.evidence_list[i] for i in lowest_indices]
            else:
                baseline_wins = None
                baseline_evid = None

            # Stage 2: LLM reasoning on suspicious windows only
            try:
                filtered_windows = self.windows[suspicious_indices]
                filtered_evidence = [self.evidence_list[i] for i in suspicious_indices]

                llm_scores = self.agent.analyze_windows(
                    filtered_windows, filtered_evidence,
                    baseline_windows=baseline_wins,
                    baseline_evidence=baseline_evid
                )

                # Merge: LLM scores for suspicious, normalize statistical for rest
                self.llm_scores = np.zeros(N)
                # Normalize statistical scores to 0-1 range for non-suspicious
                s_min = subsequence_scores.min()
                s_max = subsequence_scores.max()
                s_range = s_max - s_min if s_max > s_min else 1.0
                normalized_stat = (subsequence_scores - s_min) / s_range

                # Non-suspicious windows: use normalized stat score (capped at 0.3)
                scores_to_pool = normalized_stat * 0.3

                # Suspicious windows: use LLM confidence score
                for j, orig_idx in enumerate(suspicious_indices):
                    if j < len(llm_scores):
                        scores_to_pool[orig_idx] = llm_scores[j]
                    self.llm_scores[orig_idx] = scores_to_pool[orig_idx]

            except Exception as e:
                print(f"  Warning: LLM scoring failed, falling back to heuristic: {e}")
                scores_to_pool = subsequence_scores

        # Convert window scores to point-wise via average pooling
        point_scores = np.zeros(original_length)
        counts = np.zeros(original_length)

        for i, s in enumerate(scores_to_pool):
            start = i * stride
            end = min(start + window_size, original_length)
            point_scores[start:end] += s
            counts[start:end] += 1

        point_scores = point_scores / np.maximum(counts, 1)

        return point_scores

    def get_llm_results(self) -> Optional[List[Dict]]:
        """Get LLM analysis results from last score() call."""
        return self.agent.get_results() if self.agent else None

    def get_call_count(self) -> int:
        """Get total LLM API calls made."""
        return self.agent.get_call_count() if self.agent else 0


class LLMRangeDetectionScoring(ScoringMethod):
    """AnoAgent-style LLM range detection scoring.

    Instead of pooling window-level confidence scores, sends the full
    deseasonalized series to the LLM and asks for anomaly ranges directly.
    This produces precise anomaly boundaries and constrained detection count.

    Pipeline integration:
    - The orchestrator calls set_series_context() with the full series
    - The orchestrator calls set_evidence_context() with evidence (optional)
    - score() sends one LLM call per series and returns binary point scores
    """

    def __init__(
        self,
        backend_type: str = "azure_openai",
        temperature: Optional[float] = None,
        max_anomaly_ratio: float = 0.01,
        min_anomalies: int = 3,
        use_evidence_hints: bool = True,
        use_deseasonalized: bool = True,
        **backend_kwargs
    ):
        """Initialize LLM range detection scoring.

        Args:
            backend_type: "azure_openai", "gemini", or "claude"
            temperature: LLM temperature (default: 0.0)
            max_anomaly_ratio: Max fraction of series length used to compute
                the anomaly count constraint. Default 0.01 per AnoAgent.
            min_anomalies: Minimum anomaly count constraint. Default 3.
            use_evidence_hints: If True, include evidence summary in prompt
            use_deseasonalized: If True, use deseasonalized series;
                if False, use raw series
            **backend_kwargs: Passed to backend constructor
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from llm import create_backend

        self.backend = create_backend(backend_type, **backend_kwargs)
        self.temperature = temperature or 0.0
        self.max_anomaly_ratio = max_anomaly_ratio
        self.min_anomalies = min_anomalies
        self.use_evidence_hints = use_evidence_hints
        self.use_deseasonalized = use_deseasonalized

        # State injected by orchestrator
        self.full_series = None
        self.deseasonalized_series = None
        self.evidence_list = None
        self.windows = None

        # Results
        self.last_ranges = None
        self.call_count = 0

    def set_series_context(
        self,
        full_series: np.ndarray,
        deseasonalized_series: Optional[np.ndarray] = None
    ) -> None:
        """Inject full series from Step 1 processor.

        Called by the orchestrator between Step 1 and Step 3.
        """
        self.full_series = full_series
        self.deseasonalized_series = deseasonalized_series

    def set_evidence_context(
        self,
        evidence_list: List[Dict],
        windows: np.ndarray
    ) -> None:
        """Inject evidence from Step 2 (same interface as LLMReasoningScoring)."""
        self.evidence_list = evidence_list
        self.windows = windows

    def score(
        self,
        subsequence_scores: np.ndarray,
        window_size: int,
        stride: int,
        original_length: int
    ) -> np.ndarray:
        """Score using LLM range detection on full series.

        Sends the full (deseasonalized) series to the LLM in one call.
        LLM returns anomaly ranges which are directly converted to
        point-level binary scores.
        """
        from llm.range_prompt_builder import (
            RANGE_DETECTION_SYSTEM_PROMPT,
            build_range_detection_prompt,
            build_evidence_summary,
        )
        from llm.range_output_parser import parse_range_output, ranges_to_point_scores

        # Choose which series to send
        if self.use_deseasonalized and self.deseasonalized_series is not None:
            series_to_send = self.deseasonalized_series
            print(f"  LLM Range Detection: using deseasonalized series "
                  f"({len(series_to_send)} points)")
        elif self.full_series is not None:
            series_to_send = self.full_series
            print(f"  LLM Range Detection: using raw series "
                  f"({len(series_to_send)} points)")
        else:
            # Fallback: reconstruct from windows
            series_to_send = self._reconstruct_from_windows(
                stride, original_length
            )
            print(f"  LLM Range Detection: reconstructed from windows "
                  f"({len(series_to_send)} points)")

        # Trim or pad to match original_length
        if len(series_to_send) > original_length:
            series_to_send = series_to_send[:original_length]
        elif len(series_to_send) < original_length:
            padded = np.zeros(original_length)
            padded[:len(series_to_send)] = series_to_send
            series_to_send = padded

        # Compute anomaly count constraint (AnoAgent formula)
        max_anomalies = max(
            self.min_anomalies,
            int(original_length * self.max_anomaly_ratio)
        )

        # Build evidence summary if available and enabled
        evidence_summary = None
        if (self.use_evidence_hints
                and subsequence_scores is not None
                and len(subsequence_scores) > 0):
            evidence_summary = build_evidence_summary(
                subsequence_scores, window_size, stride, top_k=5
            )

        # Build prompt
        system_prompt = RANGE_DETECTION_SYSTEM_PROMPT.format(
            max_anomalies=max_anomalies
        )
        user_prompt = build_range_detection_prompt(
            series=series_to_send,
            max_anomalies=max_anomalies,
            evidence_summary=evidence_summary,
        )

        # Call LLM (single call per series)
        try:
            raw_response = self.backend.generate_with_retry(
                system_prompt, user_prompt, self.temperature
            )
            self.call_count += 1

            # Parse ranges
            parsed = parse_range_output(raw_response)
            anomaly_ranges = parsed.get("anomalies", [])
            self.last_ranges = anomaly_ranges

            if parsed.get("parse_error"):
                print(f"  Warning: parse issue: {parsed['parse_error']}")

            print(f"  LLM returned {len(anomaly_ranges)} anomaly ranges "
                  f"(max allowed: {max_anomalies})")

            # Convert ranges to point scores
            point_scores = ranges_to_point_scores(anomaly_ranges, original_length)

            n_flagged = int(point_scores.sum())
            pct_flagged = 100 * n_flagged / original_length
            print(f"  Flagged {n_flagged}/{original_length} points "
                  f"({pct_flagged:.1f}%)")

            return point_scores

        except Exception as e:
            print(f"  Warning: LLM range detection failed: {e}")
            print(f"  Falling back to evidence-based scoring")
            return self._fallback_scoring(
                subsequence_scores, window_size, stride, original_length
            )

    def _reconstruct_from_windows(
        self,
        stride: int,
        original_length: int,
    ) -> np.ndarray:
        """Reconstruct series from windows when full series not available."""
        if self.windows is not None:
            N = self.windows.shape[0]
            W = self.windows.shape[1]
            T = (N - 1) * stride + W
            series = np.zeros(T)
            counts = np.zeros(T)
            for i in range(N):
                start = i * stride
                vals = (self.windows[i, :, 0] if self.windows.ndim == 3
                        else self.windows[i])
                series[start:start + W] += vals
                counts[start:start + W] += 1
            return series / np.maximum(counts, 1)
        return np.zeros(original_length)

    def _fallback_scoring(
        self,
        subsequence_scores: np.ndarray,
        window_size: int,
        stride: int,
        original_length: int,
    ) -> np.ndarray:
        """Fallback: average pooling of evidence scores."""
        point_scores = np.zeros(original_length)
        counts = np.zeros(original_length)
        for i, s in enumerate(subsequence_scores):
            start = i * stride
            end = min(start + window_size, original_length)
            point_scores[start:end] += s
            counts[start:end] += 1
        return point_scores / np.maximum(counts, 1)

    def get_last_ranges(self) -> Optional[List[Dict]]:
        """Get anomaly ranges from last score() call."""
        return self.last_ranges

    def get_call_count(self) -> int:
        """Get total LLM API calls made."""
        return self.call_count
