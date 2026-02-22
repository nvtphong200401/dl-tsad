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
    """LLM-based scoring that replaces heuristic aggregation with reasoning.

    Uses an LLM to analyze statistical evidence per window and produce
    confidence scores. Falls back to max pooling of subsequence_scores
    if LLM analysis is not available.

    The orchestrator injects evidence via set_evidence_context() before
    calling score().
    """

    def __init__(
        self,
        backend_type: str = "azure_openai",
        batch_size: int = 10,
        temperature: float = 0.0,
        **backend_kwargs
    ):
        """Initialize LLM scoring.

        Args:
            backend_type: "azure_openai", "gemini", or "claude"
            batch_size: Windows per LLM call (default: 10)
            temperature: LLM temperature (default: 0.0)
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
        """Score using LLM reasoning, with fallback to max pooling.

        If evidence is available, the LLM analyzes each window and
        produces confidence scores that replace the heuristic scores.
        Otherwise, falls back to standard max pooling.

        Args:
            subsequence_scores: Window-level scores from Step 2 (N,)
            window_size: Size of each window
            stride: Stride between windows
            original_length: Length of original time series

        Returns:
            Point-wise scores (T,)
        """
        # If evidence available, use LLM scoring
        if self.evidence_list is not None and self.windows is not None:
            try:
                self.llm_scores = self.agent.analyze_windows(
                    self.windows, self.evidence_list
                )
                # Use LLM confidence scores for pooling
                scores_to_pool = self.llm_scores
            except Exception as e:
                print(f"  Warning: LLM scoring failed, falling back to heuristic: {e}")
                scores_to_pool = subsequence_scores
        else:
            scores_to_pool = subsequence_scores

        # Convert window scores to point-wise via max pooling
        point_scores = np.zeros(original_length)
        for i, s in enumerate(scores_to_pool):
            start = i * stride
            end = min(start + window_size, original_length)
            point_scores[start:end] = np.maximum(point_scores[start:end], s)

        return point_scores

    def get_llm_results(self) -> Optional[List[Dict]]:
        """Get LLM analysis results from last score() call."""
        return self.agent.get_results() if self.agent else None

    def get_call_count(self) -> int:
        """Get total LLM API calls made."""
        return self.agent.get_call_count() if self.agent else 0
