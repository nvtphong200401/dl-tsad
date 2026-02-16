#!/usr/bin/env python3
"""
Best TSAD - Train SOTA Models on Kaggle
========================================

Usage on Kaggle:
    1. Clone repo:     !git clone https://github.com/nvtphong200401/dl-tsad.git
    2. Run script:     !python dl-tsad/best-tsad/kaggle_train.py --category point

    Or train all categories:
                       !python dl-tsad/best-tsad/kaggle_train.py --all

Prerequisites:
    - Upload AnomLLM synthetic data as a Kaggle Dataset
    - Enable GPU: Settings -> Accelerator -> GPU T4 x2
    - pip install scikit-learn pyyaml pandas
"""

import argparse
import os
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Path setup — resolve the repo root so imports work regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # .../best-tsad/
REPO_ROOT = SCRIPT_DIR                                 # best-tsad IS the package root

# Remove any shadowing empty src/ in /kaggle/working/
for shadow in ["/kaggle/working/src", "/kaggle/working/configs"]:
    if os.path.isdir(shadow) and not os.path.exists(os.path.join(shadow, "__init__.py")):
        import shutil
        shutil.rmtree(shadow, ignore_errors=True)
        print(f"[cleanup] Removed shadowing directory: {shadow}")

# Ensure our repo root is first on sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.anomllm_loader import (
    load_anomllm_category,
    get_all_categories,
)
from src.utils.config_factory import load_config, build_pipeline_from_config
from src.evaluation.evaluator import Evaluator


# ---------------------------------------------------------------------------
# Default paths (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "/kaggle/working/dl-tsad/best-tsad/src/data/synthetic"
DEFAULT_OUTPUT_DIR = "/kaggle/working"
CONFIG_DIR = REPO_ROOT / "configs" / "pipelines"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def print_banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} | Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def train_model(
    model_name: str,
    config_file: str,
    dataset,
    device: str,
    output_dir: str,
    category: str,
    max_train: int = 50_000,
    max_test: int = 30_000,
    gpu_overrides: dict = None,
):
    """Train a single model, evaluate, and save artifacts.

    Returns:
        dict with evaluation metrics, or None on failure.
    """
    print_banner(f"Training {model_name} on '{category}'")

    # --- Load config ---
    config_path = str(CONFIG_DIR / config_file)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        return None

    config = load_config(config_path)

    # --- GPU overrides ---
    config["data_processing"]["params"]["device"] = device
    if gpu_overrides:
        for key, value in gpu_overrides.items():
            config["data_processing"]["params"][key] = value

    # --- Build pipeline ---
    print("Building pipeline...")
    pipeline = build_pipeline_from_config(config)

    # --- Prepare data (cap size for memory) ---
    X_train = dataset.X_train[:max_train]
    y_train = dataset.y_train[:max_train]
    X_test = dataset.X_test[:max_test]
    y_test = dataset.y_test[:max_test]

    # --- Train ---
    print(f"Training on {len(X_train):,} samples...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # --- Evaluate ---
    print("Evaluating...")
    result = pipeline.predict(X_test, y_test)
    evaluator = Evaluator()
    ev = evaluator.evaluate(y_test, result.predictions, result.point_scores)

    print(f"  F1:        {ev.f1:.4f}")
    print(f"  Precision: {ev.precision:.4f}")
    print(f"  Recall:    {ev.recall:.4f}")
    print(f"  PA-F1:     {ev.pa_f1:.4f}")
    if ev.vus_pr is not None:
        print(f"  VUS-PR:    {ev.vus_pr:.4f}")

    # --- Save model weights ---
    model_path = os.path.join(output_dir, f"{model_name}_model_{category}.pth")
    torch.save(pipeline.data_processor.model.state_dict(), model_path)

    # --- Save metadata ---
    metadata = {
        "category": category,
        "model": model_name,
        "config": config["data_processing"]["params"],
        "window_size": config["data_processing"]["window_size"],
        "input_dim": pipeline.data_processor.input_dim,
        "f1": ev.f1,
        "precision": ev.precision,
        "recall": ev.recall,
        "pa_f1": ev.pa_f1,
        "vus_pr": ev.vus_pr,
        "train_time_s": train_time,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    meta_path = os.path.join(output_dir, f"{model_name}_model_{category}_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Saved: {model_path}")
    print(f"Saved: {meta_path}")

    return {
        "Model": model_name,
        "Category": category,
        "F1": ev.f1,
        "Precision": ev.precision,
        "Recall": ev.recall,
        "PA-F1": ev.pa_f1,
        "VUS-PR": ev.vus_pr,
        "Train Time (s)": round(train_time, 1),
    }


# ---------------------------------------------------------------------------
# Model definitions (name, config file, GPU overrides)
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name": "aer",
        "config": "aer_pipeline.yaml",
        "overrides": {"hidden_dim": 128, "epochs": 50},
    },
    {
        "name": "anomaly_transformer",
        "config": "transformer_pipeline.yaml",
        "overrides": {"d_model": 512},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train SOTA anomaly detection models on Kaggle GPU"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="point",
        help="AnomLLM category to train on (default: point)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on ALL categories",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["aer", "anomaly_transformer"],
        default=None,
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to AnomLLM synthetic data (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save trained models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=50_000,
        help="Max training samples (default: 50000)",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=30_000,
        help="Max test samples (default: 30000)",
    )
    args = parser.parse_args()

    # Resolve categories
    categories = get_all_categories() if args.all else [args.category]

    # Resolve models
    selected = MODELS
    if args.models:
        selected = [m for m in MODELS if m["name"] in args.models]

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    all_results = []

    for cat in categories:
        print_banner(f"Loading category: {cat}")
        try:
            dataset = load_anomllm_category(cat, base_path=args.data_path)
        except Exception as e:
            print(f"[ERROR] Failed to load '{cat}': {e}")
            continue

        print(f"  Train: {len(dataset.X_train):,}  |  Val: {len(dataset.X_val):,}  |  Test: {len(dataset.X_test):,}")
        print(f"  Test anomalies: {int(dataset.y_test.sum()):,}")

        for model_def in selected:
            result = train_model(
                model_name=model_def["name"],
                config_file=model_def["config"],
                dataset=dataset,
                device=device,
                output_dir=args.output_dir,
                category=cat,
                max_train=args.max_train,
                max_test=args.max_test,
                gpu_overrides=model_def["overrides"],
            )
            if result:
                all_results.append(result)

    # --- Summary ---
    if all_results:
        print_banner("RESULTS SUMMARY")
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False))

        csv_path = os.path.join(args.output_dir, "training_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\n[WARNING] No models were trained successfully.")

    print("\nDone! Download models from the Output tab on Kaggle.")


if __name__ == "__main__":
    main()
