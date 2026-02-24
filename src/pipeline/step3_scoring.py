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
