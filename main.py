"""
Main entry point for the Classical vs Deep Learning License Plate Detection project.

Usage:
    python main.py <command>

Commands:
    prepare-data        Extract HOG features from training images
    train-svm-linear    Train SVM with linear-only grid search
    train-svm-full      Train SVM with full grid search (linear + RBF)
    train-yolo          Train YOLOv8n detector
    eval-svm-linear     Qualitative analysis - SVM linear model
    eval-svm-full       Qualitative analysis - SVM full (best) model
    eval-yolo           Qualitative analysis - YOLO
    eval-all            Run all three qualitative analyses
"""

import os
import sys
import cv2
import time
import shutil
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Classical ML imports
from src.classical.hog_features import HOGFeatureExtractor
from src.classical.svm_classifier import PlateClassification
from src.classical.training_data_preparation import prepare_dataset
from src.classical.train_svm import load_features, generate_report as generate_svm_report

# Deep Learning imports
from src.deep.train_yolo import YoloTrainConfig
from src.deep.train_yolo import generate_report as generate_yolo_report

# Eval imports
from src.evaluation.metrics import classification_metrics
from src.evaluation.svm_qualitative_analysis import collect_predictions, plot_grid
from src.evaluation.yolo_qualitative_analysis import (
    match_predictions,
    draw_boxes_on_image,
    plot_image_grid,
)

# Just some utils
from src.common.utils import parse_yolo_label


class Config:
    TRAIN_IMAGES    = "data/raw/train/images"
    TRAIN_LABELS    = "data/raw/train/labels"
    TEST_IMAGES     = "data/raw/test/images"
    TEST_LABELS     = "data/raw/test/labels"
    FEATURES_DIR    = "data/features"
    DATA_YAML       = "data/raw/data.yaml"

    SVM_LINEAR      = "models/svm_plate_linear.joblib"
    SVM_FULL        = "models/svm_plate_rbf.joblib"
    YOLO_MODEL      = "models/yolo_plate.pt"
    YOLO_BASE       = "src/deep/yolov8n.pt"

    OUT_SVM_LINEAR  = "outputs/svm_linear"
    OUT_SVM_FULL    = "outputs/svm_full"
    OUT_YOLO        = "outputs/yolo"
    OUT_QUAL_LINEAR = "outputs/qualitative_linear"
    OUT_QUAL_FULL   = "outputs/qualitative_full"
    OUT_QUAL_YOLO   = "outputs/qualitative_yolo"

    MAX_EVAL_IMAGES = 200
    NEG_PER_IMAGE   = 5
    SEED            = 42
    HOG_TARGET_SIZE = (64, 128)


# Parameters for SVM grid, linear is to save time.
# Full is to do full grid-search (not exactly cross-validation, if we are technical)
LINEAR_GRID = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear"],
}

FULL_GRID = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"],
}


def prepare_data():
    """Extract HOG features from training images (positive + negative crops)."""
    prepare_dataset(
        images_dir=Config.TRAIN_IMAGES,
        labels_dir=Config.TRAIN_LABELS,
        output_dir=Config.FEATURES_DIR,
        neg_per_image=Config.NEG_PER_IMAGE,
        seed=Config.SEED,
    )

def train_svm(param_grid: dict, model_path: str, output_dir: str):
    """
    Train an SVM classifier with the given param grid.
    Loads features, runs grid search, evaluates, saves model and report.
    """
    print(f"Loading features from {Config.FEATURES_DIR} ...")
    X_train, y_train, X_val, y_val = load_features(Config.FEATURES_DIR)

    if X_val is None:
        print("No validation set found, splitting 20% from training data.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=Config.SEED, stratify=y_train,
        )

    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val:   {X_val.shape[0]} samples")
    class_balance = float(np.mean(y_train))
    print(f"Class balance (train): {class_balance:.2%} positive")

    clf = PlateClassification()
    print(f"\nRunning grid search ...")
    result = clf.grid_search(X_train, y_train, param_grid=param_grid)
    best_params = result["best_params"]
    best_cv_score = result["best_score"]
    print(f"Best params: {best_params}")
    print(f"Best CV F1:  {best_cv_score:.4f}")

    print("\n--- Validation results ---")
    preds, scores = clf.predict(X_val)
    metrics = classification_metrics(y_val, preds)

    clf.save(model_path)
    print(f"\nModel saved to {model_path}")

    generate_svm_report(
        output_dir=output_dir,
        metrics=metrics,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        class_balance=class_balance,
        best_params=best_params,
        best_cv_score=best_cv_score,
        model_path=model_path,
    )


def train_svm_linear():
    train_svm(LINEAR_GRID, Config.SVM_LINEAR, Config.OUT_SVM_LINEAR)


def train_svm_full():
    train_svm(FULL_GRID, Config.SVM_FULL, Config.OUT_SVM_FULL)


# ======================================================================
# YOLO training
# ======================================================================

def train_yolo(yolo_config: YoloTrainConfig):
    """Train YOLOv8n and generate a report."""
    print("Initializing YOLOv8n (nano) model for training...")
    model = YOLO(Config.YOLO_BASE)

    data_path = str(Path(Config.DATA_YAML).resolve())
    train_args = {
        "data": data_path,
        "epochs": yolo_config.epochs,
        "imgsz": yolo_config.imgsz,
        "batch": yolo_config.batch,
        "workers": yolo_config.workers,
        "device": yolo_config.device,
        "name": yolo_config.name,
        "project": yolo_config.project,
        "save": True,
        "exist_ok": True,
        "verbose": False,
    }

    start = time.time()

    # Training function --for DEBUG-- (misha put breakpoint after this - cuz of shutil2 bug)
    results = model.train(**train_args)
    training_time = time.time() - start

    best_model_src = Path(results.save_dir) / "weights" / "best.pt"
    model_dest = Path(Config.YOLO_MODEL)
    model_dest.parent.mkdir(parents=True, exist_ok=True)

    if best_model_src.exists():
        shutil.copy2(best_model_src, model_dest)
        print(f"\nBest model copied to: {model_dest}")

    print("\nRunning validation on test set...")
    best_model = YOLO(str(best_model_src), task="detect")
    val_results = best_model.val(data=data_path, split="test")

    metrics = {}
    if val_results and val_results.box:
        metrics["mAP50"] = val_results.box.map50
        metrics["mAP50-95"] = val_results.box.map
        metrics["Precision"] = val_results.box.mp
        metrics["Recall"] = val_results.box.mr

    generate_yolo_report(
        output_dir=Config.OUT_YOLO,
        metrics=metrics,
        train_args=train_args,
        model_path=str(model_dest),
        training_time=training_time,
    )

    print(f"\nTraining complete! Time: {training_time / 60:.1f} minutes")


# ======================================================================
# SVM qualitative analysis
# ======================================================================

def eval_svm(model_path: str, output_dir: str, label: str):
    """
    Run qualitative analysis on an SVM model.
    Loads the model, runs on test crops, generates TP/FN/FP grids and report.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading SVM model ({label}) from {model_path} ...")
    clf = PlateClassification.load(model_path)
    extractor = HOGFeatureExtractor(target_size=Config.HOG_TARGET_SIZE)

    print(f"Running classifier on crops from {Config.TEST_IMAGES} ...")
    results = collect_predictions(
        Config.TEST_IMAGES, Config.TEST_LABELS, clf, extractor,
        max_images=Config.MAX_EVAL_IMAGES, seed=Config.SEED,
    )

    tp = [r for r in results if r["label"] == 1 and r["pred"] == 1]
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]
    tn = [r for r in results if r["label"] == 0 and r["pred"] == 0]

    print(f"\nResults on {len(results)} crops:")
    print(f"  True Positives:  {len(tp)}")
    print(f"  False Negatives: {len(fn)}")
    print(f"  False Positives: {len(fp)}")
    print(f"  True Negatives:  {len(tn)}")

    tp_sorted = sorted(tp, key=lambda r: r["score"])
    fn_sorted = sorted(fn, key=lambda r: r["score"])
    fp_sorted = sorted(fp, key=lambda r: r["score"], reverse=True)

    print("\nGenerating visualizations ...")
    plot_grid(
        [r["crop"] for r in tp_sorted[:15]],
        [f"score={r['score']:.2f}" for r in tp_sorted[:15]],
        f"True Positives - Lowest Confidence ({len(tp)} total)",
        out / "true_positives_low_confidence.png",
    )
    plot_grid(
        [r["crop"] for r in fn_sorted[:15]],
        [f"score={r['score']:.2f}\n{r['img_name'][:25]}" for r in fn_sorted[:15]],
        f"False Negatives - Missed Plates ({len(fn)} total)",
        out / "false_negatives.png",
    )
    plot_grid(
        [r["crop"] for r in fp_sorted[:15]],
        [f"score={r['score']:.2f}\n{r['img_name'][:25]}" for r in fp_sorted[:15]],
        f"False Positives - Background Called Plate ({len(fp)} total)",
        out / "false_positives.png",
    )

    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    report_lines = [
        f"# Qualitative Analysis Report: SVM ({label})",
        "",
        f"> Dataset: `{Config.TEST_IMAGES}`",
        f"> Model: `{model_path}`",
        f"> Images analyzed: {Config.MAX_EVAL_IMAGES}",
        "",
        "## Summary",
        "",
        "The SVM is evaluated on pre-cropped patches, each plate region and several",
        "random background regions are extracted, passed through HOG, and classified.",
        "This measures the quality of the HOG + SVM combination in isolation,",
        "independent of the sliding window localization step.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total crops | {len(results)} |",
        f"| Positive crops (plates) | {len(tp) + len(fn)} |",
        f"| Negative crops (background) | {len(fp) + len(tn)} |",
        f"| True Positives | {len(tp)} |",
        f"| False Negatives | {len(fn)} |",
        f"| False Positives | {len(fp)} |",
        f"| True Negatives | {len(tn)} |",
        f"| Precision | {precision:.4f} |",
        f"| Recall | {recall:.4f} |",
        f"| F1 | {f1:.4f} |",
        "",
        "## Visualizations",
        "",
        "### True Positives (lowest confidence)",
        "These are plates the model correctly identified, but with the lowest scores.",
        "They show the boundary of what the model considers a plate, the cases",
        "where the HOG gradient pattern was just barely 'plate-like' enough to pass.",
        "",
        "![True Positives](true_positives_low_confidence.png)",
        "",
        "### False Negatives (missed plates)",
        "These are actual plates that the model classified as background.",
        "Look for patterns: small plates, blur, occlusion, unusual angles.",
        "Some may be too degraded to classify even by human eye, those",
        "represent a data quality issue rather than a model failure.",
        "",
        "![False Negatives](false_negatives.png)",
        "",
        "### False Positives (background called plate)",
        "These are background patches the model mistakenly called plates.",
        "Look for patterns: rectangular shapes, text-like textures, high contrast",
        "edges, anything that produces a plate-like HOG gradient signature.",
        "",
        "![False Positives](false_positives.png)",
    ]
    (out / "qualitative_report.md").write_text("\n".join(report_lines))
    print(f"\nReport saved to: {out / 'qualitative_report.md'}")


def eval_svm_linear():
    eval_svm(Config.SVM_LINEAR, Config.OUT_QUAL_LINEAR, "Linear")


def eval_svm_full():
    eval_svm(Config.SVM_FULL, Config.OUT_QUAL_FULL, "Full")


# ======================================================================
# YOLO qualitative analysis
# ======================================================================

def eval_yolo():
    """Run qualitative analysis on the YOLO model."""
    out = Path(Config.OUT_QUAL_YOLO)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model from {Config.YOLO_MODEL} ...")
    model = YOLO(Config.YOLO_MODEL, task="detect")

    img_names = sorted([
        f for f in os.listdir(Config.TEST_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    random.seed(Config.SEED)
    if len(img_names) > Config.MAX_EVAL_IMAGES:
        img_names = random.sample(img_names, Config.MAX_EVAL_IMAGES)

    print(f"Running YOLO on {len(img_names)} images ...")

    all_tp_images, all_fn_images, all_fp_images = [], [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for img_name in img_names:
        img_path = os.path.join(Config.TEST_IMAGES, img_name)
        label_path = os.path.join(Config.TEST_LABELS, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = parse_yolo_label(label_path, img_w, img_h) if os.path.exists(label_path) else []

        results = model.predict(img, conf=0.25, verbose=False)
        pred_boxes = []
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                pred_boxes.append((x1, y1, x2, y2, conf))

        tp_pairs, fp_boxes, fn_boxes = match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += len(tp_pairs)
        total_fp += len(fp_boxes)
        total_fn += len(fn_boxes)

        # short_name = img_name[:30]
        #
        # if fn_boxes:
        #     vis = draw_boxes_on_image(img, gt_boxes=fn_boxes, pred_boxes=pred_boxes)
        #     all_fn_images.append((vis, f"{short_name}\n{len(fn_boxes)} missed"))
        #
        # if fp_boxes:
        #     vis = draw_boxes_on_image(img, gt_boxes=gt_boxes, pred_boxes=fp_boxes)
        #     all_fp_images.append((vis, f"{short_name}\n{len(fp_boxes)} false det"))
        #
        # for pred, gt, iou in tp_pairs:
        #     vis = draw_boxes_on_image(img, gt_boxes=[gt], pred_boxes=[pred])
        #     all_tp_images.append((vis, f"{short_name}\nconf={pred[4]:.2f} IoU={iou:.2f}"))

        short_name = img_name[:30]
        tp_gt = [gt for _, gt, _ in tp_pairs]
        tp_pred = [pred for pred, _, _ in tp_pairs]

        # FN grid: show full context of images with misses
        if fn_boxes:
            vis = draw_boxes_on_image(img, tp_gt_boxes=tp_gt, fn_boxes=fn_boxes,
                                      fp_boxes=fp_boxes, tp_pred_boxes=tp_pred)
            all_fn_images.append((vis, f"{short_name}\n{len(fn_boxes)} missed"))

        # FP grid: show full context of images with false detections
        if fp_boxes:
            vis = draw_boxes_on_image(img, tp_gt_boxes=tp_gt, fn_boxes=fn_boxes,
                                      fp_boxes=fp_boxes, tp_pred_boxes=tp_pred)
            all_fp_images.append((vis, f"{short_name}\n{len(fp_boxes)} false det"))

        # TP grid: one image per TP, just that pair
        for pred, gt, iou in tp_pairs:
            vis = draw_boxes_on_image(img, tp_gt_boxes=[gt], tp_pred_boxes=[pred])
            all_tp_images.append((vis, f"{short_name}\nconf={pred[4]:.2f} IoU={iou:.2f}"))

    all_tp_images.sort(key=lambda x: float(x[1].split("conf=")[1].split(" ")[0]))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResults on {len(img_names)} images:")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    print("\nGenerating visualizations ...")
    plot_image_grid(
        [img for img, _ in all_tp_images[:9]],
        [t for _, t in all_tp_images[:9]],
        f"YOLO True Positives, Lowest Confidence ({total_tp} total)",
        out / "yolo_true_positives.png",
    )
    plot_image_grid(
        [img for img, _ in all_fn_images[:9]],
        [t for _, t in all_fn_images[:9]],
        f"YOLO False Negatives, Missed Plates ({total_fn} total)",
        out / "yolo_false_negatives.png",
    )
    plot_image_grid(
        [img for img, _ in all_fp_images[:9]],
        [t for _, t in all_fp_images[:9]],
        f"YOLO False Positives, Spurious Detections ({total_fp} total)",
        out / "yolo_false_positives.png",
    )

    report_lines = [
        "# YOLO Qualitative Analysis Report",
        "",
        f"> Dataset: `{Config.TEST_IMAGES}`",
        f"> Model: `{Config.YOLO_MODEL}`",
        f"> Images analyzed: {len(img_names)}",
        f"> IoU threshold: 0.5",
        f"> Confidence threshold: 0.25",
        "",
        "## Summary",
        "",
        "Unlike the SVM qualitative analysis which evaluates on cropped patches,",
        "this analysis runs full-image detection YOLO must both locate and classify",
        "the plate in a single pass. Predictions are matched to ground truth using IoU.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Images analyzed | {len(img_names)} |",
        f"| True Positives | {total_tp} |",
        f"| False Positives | {total_fp} |",
        f"| False Negatives | {total_fn} |",
        f"| Precision | {precision:.4f} |",
        f"| Recall | {recall:.4f} |",
        f"| F1 | {f1:.4f} |",
        "",
        "## Visualizations",
        "",
        "In all images below: **green boxes** = ground truth, **red boxes** = YOLO predictions.",
        "",
        "### True Positives (lowest confidence)",
        "These are plates YOLO detected correctly, but with the lowest confidence scores.",
        "They represent the edge of what the model considers a plate the cases where",
        "it was least sure but still got it right. Look for patterns: distant plates,",
        "unusual angles, or partially visible plates that made the model hesitate.",
        "",
        "![True Positives](yolo_true_positives.png)",
        "",
        "### False Negatives (missed plates)",
        "These are actual plates that YOLO failed to detect entirely.",
        "Compare these with the SVM false negatives to see whether both models",
        "struggle with the same cases (e.g. tiny plates, heavy blur) or whether",
        "YOLO handles some failure modes that the classical pipeline cannot.",
        "",
        "![False Negatives](yolo_false_negatives.png)",
        "",
        "### False Positives (spurious detections)",
        "These are detections that don't match any ground truth plate.",
        "Look for what confused the model: rectangular signs, text on buildings,",
        "reflective surfaces, or other plate-like patterns in the scene.",
        "",
        "![False Positives](yolo_false_positives.png)",
    ]
    (out / "yolo_qualitative_report.md").write_text("\n".join(report_lines))
    print(f"\nReport saved to: {out / 'yolo_qualitative_report.md'}")


def eval_all():
    """Run all three qualitative analyses."""
    eval_svm_linear()
    print()
    eval_svm_full()
    print()
    eval_yolo()

def train_yolo_factory():
    # Misha - if you need to reduce workload
    # Ex: train_yolo(YoloTrainConfig(batch=8, workers=2))
    train_yolo(YoloTrainConfig())


COMMANDS = {
    "prepare-data":     (prepare_data,      "Extract HOG features from training images"),
    "train-svm-linear": (train_svm_linear,  "Train SVM - linear kernel only"),
    "train-svm-full":   (train_svm_full,    "Train SVM - full grid (linear + RBF)"),
    "train-yolo":       (train_yolo_factory,        "Train YOLOv8n detector"),
    "eval-svm-linear":  (eval_svm_linear,   "Qualitative analysis - SVM linear"),
    "eval-svm-full":    (eval_svm_full,     "Qualitative analysis - SVM full (best)"),
    "eval-yolo":        (eval_yolo,         "Qualitative analysis - YOLO"),
    "eval-all":         (eval_all,          "Run all three qualitative analyses"),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print("\nClassical vs Deep Learning: License Plate Detection")
        print("=" * 55)
        print("\nUsage: python main.py <command>\n")
        print("Available commands:\n")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<20} {desc}")
        print()
        return

    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"\nUnknown command: '{cmd_name}'")
        print("Run 'python main.py help' to see available commands.\n")
        sys.exit(1)

    fn, _ = COMMANDS[cmd_name]
    fn()


if __name__ == "__main__":
    main()