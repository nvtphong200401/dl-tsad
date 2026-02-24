"""LLM Anomaly Detection Agent

Main agent class that orchestrates prompting, LLM calls, and output parsing.
"""

import numpy as np
from typing import Dict, List, Optional

from .backends import LLMBackend
from .prompt_builder import SYSTEM_PROMPT, build_batch_prompt, build_single_prompt
from .output_parser import parse_llm_output, extract_window_confidence


class LLMAnomalyAgent:
    """LLM-based anomaly detection agent.

    Analyzes statistical evidence using an LLM to produce anomaly
    confidence scores with explainable reasoning.

    Args:
        backend: LLM backend instance (AzureOpenAI, Gemini, or Claude)
        batch_size: Number of windows per LLM call (default: 10)
        temperature: LLM sampling temperature (default: 0.0)
    """

    def __init__(
        self,
        backend: LLMBackend,
        batch_size: int = 10,
        temperature: float = 0.0
    ):
        self.backend = backend
        self.batch_size = batch_size
        self.temperature = temperature
        self.call_count = 0
        self.llm_results = []  # Store all LLM responses

    def analyze_windows(
        self,
        windows: np.ndarray,
        evidence_list: List[Dict],
        baseline_windows: Optional[List[np.ndarray]] = None,
        baseline_evidence: Optional[List[Dict]] = None,
        progress: bool = True
    ) -> np.ndarray:
        """Analyze all windows and return confidence scores.

        Args:
            windows: Test windows (N, W, D)
            evidence_list: Evidence dicts from EvidenceBasedDetection
            baseline_windows: Optional normal windows for contrast
            baseline_evidence: Optional evidence for normal windows
            progress: Print progress info

        Returns:
            Confidence scores (N,) where higher = more anomalous
        """
        N = len(evidence_list)
        scores = np.zeros(N)
        self.llm_results = []

        # Process in batches
        n_batches = (N + self.batch_size - 1) // self.batch_size
        if progress:
            print(f"  LLM analyzing {N} windows in {n_batches} batches...")

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, N)
            batch_indices = list(range(start, end))

            batch_windows = [windows[i, :, 0] if windows.ndim == 3 else windows[i]
                             for i in batch_indices]
            batch_evidence = [evidence_list[i] for i in batch_indices]

            if progress and (batch_idx + 1) % 5 == 0:
                print(f"    Batch {batch_idx + 1}/{n_batches}...")

            try:
                result = self._analyze_batch(
                    batch_windows, batch_evidence, batch_indices,
                    baseline_windows=baseline_windows,
                    baseline_evidence=baseline_evidence
                )
                self.llm_results.append(result)

                # Extract confidence scores
                confidence_map = extract_window_confidence(result, batch_indices)
                for idx in batch_indices:
                    scores[idx] = confidence_map.get(idx, 0.0)

            except Exception as e:
                print(f"    Warning: LLM batch {batch_idx} failed: {e}")
                # Keep scores at 0.0 for failed batches
                self.llm_results.append({
                    "windows": [],
                    "parse_error": str(e)
                })

        if progress:
            n_anomalous = np.sum(scores >= 0.5)
            print(f"  LLM analysis complete: {n_anomalous}/{N} windows flagged as anomalous")

        return scores

    def analyze_single(
        self,
        window_values: np.ndarray,
        evidence: Dict,
        window_index: int = 0
    ) -> Dict:
        """Analyze a single window.

        Args:
            window_values: Window data (W,)
            evidence: Evidence dict
            window_index: Window index

        Returns:
            Parsed LLM result dict
        """
        prompt = build_single_prompt(window_values, evidence, window_index)
        raw_response = self.backend.generate_with_retry(
            SYSTEM_PROMPT, prompt, self.temperature
        )
        self.call_count += 1
        return parse_llm_output(raw_response)

    def _analyze_batch(
        self,
        windows: List[np.ndarray],
        evidence_list: List[Dict],
        window_indices: List[int],
        baseline_windows: Optional[List[np.ndarray]] = None,
        baseline_evidence: Optional[List[Dict]] = None
    ) -> Dict:
        """Analyze a batch of windows in a single LLM call."""
        prompt = build_batch_prompt(
            windows, evidence_list, window_indices,
            baseline_windows=baseline_windows,
            baseline_evidence=baseline_evidence
        )
        raw_response = self.backend.generate_with_retry(
            SYSTEM_PROMPT, prompt, self.temperature
        )
        self.call_count += 1
        return parse_llm_output(raw_response)

    def get_results(self) -> List[Dict]:
        """Get all LLM results from last analyze_windows() call."""
        return self.llm_results

    def get_call_count(self) -> int:
        """Get total number of LLM API calls made."""
        return self.call_count
