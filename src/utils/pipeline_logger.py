"""Pipeline Debug Logger

Captures per-series diagnostic information for debugging anomaly detection
pipelines. Writes JSONL files with one record per series.

Usage:
    logger = PipelineLogger("output_dir", "config_name", "category")
    # ... after each series prediction ...
    logger.log_series(series_idx, pipeline, result, y_true)
    # ... at the end ...
    logger.close()
"""

import json
import os
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime


class PipelineLogger:
    """Structured debug logger for pipeline runs."""

    def __init__(self, output_dir: str, config_name: str, category: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_{config_name}_{category}_{timestamp}.jsonl"
        self.path = os.path.join(output_dir, filename)
        self.f = open(self.path, 'w', encoding='utf-8')
        self.config_name = config_name
        self.category = category

    def log_series(
        self,
        series_idx: int,
        pipeline,
        result,
        y_true: np.ndarray,
    ) -> None:
        """Log diagnostic info for one series prediction.

        Args:
            series_idx: Index of the series in the eval set
            pipeline: The AnomalyDetectionPipeline instance
            result: PipelineResult from pipeline.predict()
            y_true: Ground truth labels (T,)
        """
        record = {
            "series_idx": series_idx,
            "config": self.config_name,
            "category": self.category,
            "series_length": len(y_true),
        }

        # Ground truth
        record["ground_truth"] = {
            "total_anomaly_points": int(y_true.sum()),
            "anomaly_ratio": float(y_true.mean()),
            "anomaly_segments": _get_segments(y_true),
        }

        # Predictions
        preds = result.predictions
        record["predictions"] = {
            "total_detected_points": int(preds.sum()),
            "detection_ratio": float(preds.mean()),
            "detected_segments": _get_segments(preds),
            "threshold": float(result.threshold),
        }

        # Per-point metrics
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        record["metrics"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        # Scoring info
        record["scores"] = {
            "subsequence_min": float(result.subsequence_scores.min()),
            "subsequence_max": float(result.subsequence_scores.max()),
            "subsequence_mean": float(result.subsequence_scores.mean()),
            "point_score_unique_count": int(len(np.unique(result.point_scores))),
        }

        # Evidence summary (top 5 windows)
        if hasattr(pipeline.detection_method, 'get_evidence'):
            evidence = pipeline.detection_method.get_evidence()
            if evidence:
                subseq = result.subsequence_scores
                top5 = np.argsort(subseq)[-5:][::-1]
                top_evidence = []
                for idx in top5:
                    if idx < len(evidence):
                        e = evidence[idx]
                        top_evidence.append({
                            "window_idx": int(idx),
                            "score": round(float(subseq[idx]), 4),
                            "mae": round(float(e.get("mae", 0)), 4),
                            "max_z": round(float(e.get("max_abs_z_score", 0)), 4),
                            "volatility": round(float(e.get("volatility_ratio", 0)), 4),
                            "kl_div": round(float(e.get("kl_divergence", 0)), 4),
                            "violation": round(float(e.get("violation_ratio", 0)), 4),
                        })
                record["top_evidence"] = top_evidence

        # Side-by-side comparison (easy to read)
        gt_segs = record["ground_truth"]["anomaly_segments"]
        det_segs = record["predictions"]["detected_segments"]
        record["comparison"] = {
            "ground_truth_ranges": gt_segs,
            "llm_detected_ranges": [],
            "overlap_assessment": "no_anomalies" if not gt_segs else "missed",
        }

        # LLM-specific info
        scoring = pipeline.scoring_method

        # LLMRangeDetectionScoring
        if hasattr(scoring, 'last_ranges') and scoring.last_ranges is not None:
            record["llm_ranges"] = scoring.last_ranges
            record["comparison"]["llm_detected_ranges"] = [
                [r["start"], r["end"]] for r in scoring.last_ranges
            ]
            record["llm_call_count"] = getattr(scoring, 'call_count', 0)

            # Assess overlap between LLM ranges and ground truth
            if gt_segs and scoring.last_ranges:
                overlaps = []
                for llm_r in scoring.last_ranges:
                    ls, le = llm_r["start"], llm_r["end"]
                    for gs in gt_segs:
                        if isinstance(gs, list) and len(gs) == 2:
                            overlap_start = max(ls, gs[0])
                            overlap_end = min(le, gs[1])
                            if overlap_start < overlap_end:
                                overlaps.append({
                                    "llm": [ls, le],
                                    "gt": gs,
                                    "overlap": [overlap_start, overlap_end],
                                    "overlap_pts": overlap_end - overlap_start,
                                })
                record["comparison"]["overlaps"] = overlaps
                if overlaps:
                    record["comparison"]["overlap_assessment"] = "partial_match"
                else:
                    record["comparison"]["overlap_assessment"] = (
                        "false_positive" if scoring.last_ranges else "missed"
                    )
            elif not gt_segs and scoring.last_ranges:
                record["comparison"]["overlap_assessment"] = "false_positive"
            elif not gt_segs and not scoring.last_ranges:
                record["comparison"]["overlap_assessment"] = "true_negative"

        # CUSUM/signal diagnostics (for range detection)
        if hasattr(pipeline.data_processor, 'get_deseasonalized_series'):
            deseas = pipeline.data_processor.get_deseasonalized_series()
            if deseas is not None:
                try:
                    from llm.range_prompt_builder import compute_derived_signals
                    cusum, rmdev = compute_derived_signals(deseas)
                    std = max(float(np.std(deseas)), 1e-10)
                    c_scaled = np.clip(cusum / (10.0 * std) * 10.0, 0.0, 10.0)

                    # Show CUSUM at key regions
                    signal_diag = {
                        "cusum_max": round(float(c_scaled.max()), 2),
                        "cusum_argmax": int(np.argmax(c_scaled)),
                    }

                    # CUSUM at LLM-flagged region
                    if hasattr(scoring, 'last_ranges') and scoring.last_ranges:
                        for r in scoring.last_ranges:
                            s, e = r["start"], min(r["end"], len(c_scaled))
                            signal_diag[f"cusum_at_llm_{s}_{e}"] = round(
                                float(c_scaled[s:e].mean()), 2
                            )

                    # CUSUM at ground truth region
                    for gs in gt_segs:
                        if isinstance(gs, list) and len(gs) == 2:
                            s, e = gs[0], min(gs[1], len(c_scaled))
                            signal_diag[f"cusum_at_gt_{s}_{e}"] = round(
                                float(c_scaled[s:e].mean()), 2
                            )

                    # CUSUM at start/middle/end for context
                    signal_diag["cusum_first100"] = round(float(c_scaled[:100].mean()), 2)
                    signal_diag["cusum_mid"] = round(float(c_scaled[400:600].mean()), 2)
                    signal_diag["cusum_last100"] = round(float(c_scaled[-100:].mean()), 2)

                    record["signal_diagnostics"] = signal_diag
                except Exception:
                    pass

        # LLMReasoningScoring
        if hasattr(scoring, 'llm_scores') and scoring.llm_scores is not None:
            record["llm_window_scores"] = {
                "nonzero_count": int(np.sum(scoring.llm_scores > 0)),
                "max_score": round(float(scoring.llm_scores.max()), 4),
            }

        # LLM results with reasoning
        if hasattr(scoring, 'get_llm_results'):
            llm_results = scoring.get_llm_results()
            if llm_results:
                reasoning_entries = []
                for batch_result in llm_results:
                    for w in batch_result.get("windows", []):
                        reasoning_entries.append({
                            "window_index": w.get("window_index"),
                            "is_anomaly": w.get("is_anomaly"),
                            "confidence": w.get("confidence"),
                            "reasoning": w.get("reasoning", ""),
                            "evidence_cited": w.get("evidence_cited", []),
                        })
                if reasoning_entries:
                    record["llm_reasoning"] = reasoning_entries

        # Execution time
        record["execution_time"] = {
            k: round(v, 3) for k, v in result.execution_time.items()
        }

        self.f.write(json.dumps(record) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
        print(f"  [DEBUG LOG] {self.path}")

    @property
    def filepath(self):
        return self.path


def _get_segments(labels: np.ndarray, max_segments: int = 20):
    """Extract anomaly segments as list of [start, end] pairs."""
    segments = []
    in_anomaly = False
    start = 0
    for i, val in enumerate(labels):
        if val == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif val == 0 and in_anomaly:
            segments.append([start, i])
            in_anomaly = False
    if in_anomaly:
        segments.append([start, len(labels)])
    if len(segments) > max_segments:
        segments = segments[:max_segments] + [["... truncated"]]
    return segments
