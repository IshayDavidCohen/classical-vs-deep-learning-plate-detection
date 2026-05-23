"""
Train YOLOv8n on the license plate detection dataset.

Fine-tunes a pretrained YOLOv8n (nano) model on the same dataset
used for the classical HOG + SVM pipeline, so both approaches
can be compared under identical conditions.

Expects a data.yaml file in YOLO format pointing to train/val/test splits.
"""

import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class YoloTrainConfig:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    workers: int = 4
    device: str = "0"
    project: str = "outputs/yolo_runs"
    name: str = "license_plate_detector"

def generate_report(
    output_dir: str,
    metrics: dict,
    train_args: dict,
    model_path: str,
    training_time: float | None = None,
):
    """
    Save a markdown report summarizing YOLO training results.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# YOLOv8n Training Report",
        "",
        f"> Generated on {timestamp}",
        "",
        "## Overview",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Model | YOLOv8n (nano) |",
        f"| Model saved to | `{model_path}` |",
        f"| Dataset | `{train_args.get('data', '')}` |",
        f"| Epochs | {train_args.get('epochs', '')} |",
        f"| Image size | {train_args.get('imgsz', '')} |",
        f"| Batch size | {train_args.get('batch', '')} |",
        f"| Device | {train_args.get('device', '')} |",
    ]

    if training_time is not None:
        minutes = training_time / 60
        lines.append(f"| Training time | {minutes:.1f} minutes |")

    lines += [""]

    # Validation metrics from YOLO results
    if metrics:
        lines += [
            "## Validation Metrics",
            "",
            "| Metric | Score |",
            "| --- | --- |",
        ]
        for key, val in metrics.items():
            if isinstance(val, float):
                lines.append(f"| {key} | {val:.4f} |")
            else:
                lines.append(f"| {key} | {val} |")
        lines += [""]

    lines += [
        "## Training Curves",
        "",
        "Training curves and additional plots are saved in the YOLO run directory.",
        "",
    ]

    report_text = "\n".join(lines)

    report_path = out / "yolo_training_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_path}")
