"""
Train the SVM classifier on HOG features.

This script demonstrates the full training workflow:
1. Load HOG features (from .npy files your teammate produces)
2. Split into train / val
3. Train SVM (with optional grid search)
4. Evaluate on validation set
5. Save the trained model
6. Generate an automatic report with confusion matrix and metrics

Run:
PYTHONPATH=. python3 -m src.classical.train_svm \
    --data-dir G:/coding/repos/classical-vs-deep-learning-plate-detection/data/features \
    --model-path models/svm_plate.joblib \
    --output-dir ../../outputs \
    --grid-search

Expects your teammate to have saved features like:
    data/features/X_train.npy   - shape (n_samples, n_features)
    data/features/y_train.npy   - shape (n_samples,)
    data/features/X_val.npy     - (optional, for evaluation)
    data/features/y_val.npy     - (optional)

If val files don't exist, a random split is made from the training data.
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.classical.svm_classifier import PlateClassification
from src.evaluation.metrics import classification_metrics


def load_features(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load feature arrays from disk."""
    data_dir = Path(data_dir)

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    X_val, y_val = None, None
    if (data_dir / "X_val.npy").exists():
        X_val = np.load(data_dir / "X_val.npy")
        y_val = np.load(data_dir / "y_val.npy")

    return X_train, y_train, X_val, y_val


def generate_report(
        output_dir: str,
        metrics: dict,
        train_shape: tuple,
        val_shape: tuple,
        class_balance: float,
        best_params: dict | None = None,
        best_cv_score: float | None = None,
        model_path: str = "",
):
    """
    Save a confusion matrix plot and a text summary to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Confusion matrix plot ----
    cm = metrics["confusion_matrix"]
    disp = ConfusionMatrixDisplay(cm, display_labels=["background", "plate"])
    disp.plot(cmap="Blues")
    plt.title("SVM Validation - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Markdown report ----
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# SVM Training Report",
        "",
        f"> Generated on {timestamp}",
        "",
        "## Overview",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Model path | `{model_path}` |",
        f"| Train samples | {train_shape[0]:,} |",
        f"| Feature dimensions | {train_shape[1]:,} |",
        f"| Validation samples | {val_shape[0]:,} |",
        f"| Class balance (train) | {class_balance:.2%} positive |",
        "",
    ]

    if best_params is not None:
        lines += [
            "## Grid Search",
            "",
            "| Parameter | Best Value |",
            "| --- | --- |",
        ]
        for key, val in best_params.items():
            clean_key = key.replace("svm__", "")
            lines.append(f"| {clean_key} | `{val}` |")
        lines += [
            f"| **Best CV F1** | **{best_cv_score:.4f}** |",
            "",
        ]

    lines += [
        "## Validation Metrics",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 Score | {metrics['f1']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "![Confusion Matrix](confusion_matrix.png)",
        "",
        "| | Predicted Background | Predicted Plate |",
        "| --- | --- | --- |",
        f"| **Actual Background** | {cm[0][0]:,} (TN) | {cm[0][1]:,} (FP) |",
        f"| **Actual Plate** | {cm[1][0]:,} (FN) | {cm[1][1]:,} (TP) |",
        "",
    ]

    report_text = "\n".join(lines)

    # Save to file
    report_path = out / "svm_training_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to:           {report_path}")
    print(f"Confusion matrix saved to: {out / 'confusion_matrix.png'}")
