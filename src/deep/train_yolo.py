import os
from pathlib import Path

from ultralytics import YOLO


"""
Train YOLOv8n on the license plate detection dataset.

Fine-tunes a pretrained YOLOv8n (nano) model on the same dataset
used for the classical HOG + SVM pipeline, so both approaches
can be compared under identical conditions.

Usage:
    python -m src.deep.train_yolo

    or with custom args:
    python -m src.deep.train_yolo \
        --data data/raw/data.yaml \
        --epochs 100 \
        --batch 8 \
        --workers 2 \
        --output-dir outputs/yolo

Expects a data.yaml file in YOLO format pointing to train/val/test splits.
"""

import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on license plate dataset")
    parser.add_argument("--data", type=str, default="data/raw/data.yaml",
                        help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (lower if OOM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--workers", type=int, default=2,
                        help="Data loader workers (lower if RAM issues)")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--model-path", type=str, default="models/yolo_plate.pt",
                        help="Where to copy the best model after training")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Where to save the training report")
    args = parser.parse_args()

    # ---- Train ----
    print("\nInitializing YOLOv8n (nano) model for training...")
    model = YOLO("yolov8n.pt")

    train_args = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "name": "license_plate_detector",
        "save": True,
        "exist_ok": True,
        "verbose": False,
        "project": "outputs/yolo_runs",
    }

    import time
    start = time.time()
    results = model.train(**train_args)
    training_time = time.time() - start

    # ---- Copy best model ----
    best_model_src = Path(results.save_dir) / "weights" / "best.pt"
    model_dest = Path(args.model_path)
    model_dest.parent.mkdir(parents=True, exist_ok=True)

    if best_model_src.exists():
        import shutil
        shutil.copy2(best_model_src, model_dest)
        print(f"\nBest model copied to: {model_dest}")
    else:
        print(f"\nWarning: best.pt not found at {best_model_src}")
        model_dest = best_model_src

    # ---- Validate on test set ----
    print("\nRunning validation on test set...")
    best_model = YOLO(str(best_model_src), task="detect")
    val_results = best_model.val(data=train_args["data"], split="test")

    metrics = {}
    if val_results and val_results.box:
        metrics["mAP50"] = val_results.box.map50
        metrics["mAP50-95"] = val_results.box.map
        metrics["Precision"] = val_results.box.mp
        metrics["Recall"] = val_results.box.mr

    # ---- Generate report ----
    generate_report(
        output_dir=args.output_dir,
        metrics=metrics,
        train_args=train_args,
        model_path=str(model_dest),
        training_time=training_time,
    )

    print(f"\nTraining complete!")
    print(f"  Model:   {model_dest}")
    print(f"  Time:    {training_time / 60:.1f} minutes")
    if metrics:
        print(f"  mAP50:   {metrics.get('mAP50', 'N/A'):.4f}")
        print(f"  mAP50-95:{metrics.get('mAP50-95', 'N/A'):.4f}")


if __name__ == "__main__":
    main()

