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
    eval-crops          Apples-to-apples crop comparison (all 3 models)
    eval-detection      Full-image detection comparison (SVM sliding window vs YOLO)
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

    best_model_src = Path(f"{results.save_dir}/weights/best.pt")
    if not best_model_src.exists():
        best_model_src = Path(f"{results.save_dir}/weights/last.pt")
    if not best_model_src.exists():
        print(f"\nError: No model weights found at {results.save_dir}/weights/")
        return

    model_dest = Path(Config.YOLO_MODEL)
    model_dest.parent.mkdir(parents=True, exist_ok=True)
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
        max_images=Config.MAX_EVAL_IMAGES, neg_per_image=Config.NEG_PER_IMAGE, seed=Config.SEED,
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
        "In all images below: **green** = GT matched, **blue** = prediction matched, "
        "**yellow** = GT missed (false negative), **red** = false detection (false positive).",
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


def eval_crops_comparison():
    """
    Apples-to-apples comparison: run SVM linear, SVM RBF, and YOLO
    on the exact same set of crops (plate + background patches from test images).

    All three models classify the same inputs, so metrics are directly comparable.
    Generates a side-by-side comparison report.
    """
    out = Path("outputs/crops_comparison")
    out.mkdir(parents=True, exist_ok=True)

    # --- Collect crops (same for all models) ---
    print("Collecting test crops ...")
    extractor = HOGFeatureExtractor(target_size=Config.HOG_TARGET_SIZE)

    # We need both the raw crops (for YOLO) and the HOG features (for SVM).
    # collect_predictions returns crops + SVM results, but we need raw crops
    # separately for YOLO. So we collect crops manually.

    img_names = sorted([
        f for f in os.listdir(Config.TEST_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    random.seed(Config.SEED)
    if len(img_names) > Config.MAX_EVAL_IMAGES:
        img_names = random.sample(img_names, Config.MAX_EVAL_IMAGES)

    crops = []  # list of (crop_bgr, label, img_name)

    for img_name in img_names:
        img_path = os.path.join(Config.TEST_IMAGES, img_name)
        label_path = os.path.join(Config.TEST_LABELS, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = parse_yolo_label(label_path, img_w, img_h)
        if not gt_boxes:
            continue

        # Positive crops
        for (xmin, ymin, xmax, ymax) in gt_boxes:
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            crops.append((crop, 1, img_name))

        # Negative crops
        from src.common.utils import sample_negative_crops
        neg_boxes = sample_negative_crops(img_h, img_w, gt_boxes, num_negatives=Config.NEG_PER_IMAGE)
        for (xmin, ymin, xmax, ymax) in neg_boxes:
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            crops.append((crop, 0, img_name))

    labels = np.array([c[1] for c in crops])
    print(f"Collected {len(crops)} crops ({np.sum(labels)} plates, {len(labels) - np.sum(labels)} background)")

    # --- Run SVM Linear ---
    print(f"\nRunning SVM Linear on {len(crops)} crops ...")
    clf_linear = PlateClassification.load(Config.SVM_LINEAR)
    linear_preds = []
    for crop_bgr, _, img_name in crops:
        try:
            feat = extractor.compute_single(crop_bgr)
            pred, _ = clf_linear.predict(feat.reshape(1, -1))
            linear_preds.append(int(pred[0]))
        except Exception as e:
            raise RuntimeError(
                f"SVM Linear crop prediction failed for image '{img_name}' "
                f"with crop shape {crop_bgr.shape}"
            ) from e
    linear_preds = np.array(linear_preds)

    # --- Run SVM RBF ---
    print(f"Running SVM RBF on {len(crops)} crops ...")
    clf_rbf = PlateClassification.load(Config.SVM_FULL)
    rbf_preds = []
    for crop_bgr, _, img_name in crops:
        try:
            feat = extractor.compute_single(crop_bgr)
            pred, _ = clf_rbf.predict(feat.reshape(1, -1))
            rbf_preds.append(int(pred[0]))
        except Exception as e:
            raise RuntimeError(
                f"SVM RBF crop prediction failed for image '{img_name}' "
                f"with crop shape {crop_bgr.shape}"
            ) from e
    rbf_preds = np.array(rbf_preds)

    # --- Run YOLO ---
    print(f"Running YOLO on {len(crops)} crops ...")
    yolo_model = YOLO(Config.YOLO_MODEL, task="detect")
    yolo_preds = []
    for crop_bgr, _, _ in crops:
        results = yolo_model.predict(crop_bgr, conf=0.25, verbose=False)
        detected = 0
        if results and len(results[0].boxes) > 0:
            detected = 1
        yolo_preds.append(detected)
    yolo_preds = np.array(yolo_preds)

    # --- Compute metrics for each ---
    def compute_metrics(y_true, y_pred):
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1}

    m_linear = compute_metrics(labels, linear_preds)
    m_rbf = compute_metrics(labels, rbf_preds)
    m_yolo = compute_metrics(labels, yolo_preds)

    # --- Print ---
    print(f"\n{'=' * 60}")
    print(f"CROPS COMPARISON - {len(crops)} identical crops")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'SVM Linear':>12} {'SVM RBF':>12} {'YOLOv8n':>12}")
    print(f"{'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12}")
    for key in ["tp", "fn", "fp", "tn", "precision", "recall", "f1"]:
        v1 = m_linear[key]
        v2 = m_rbf[key]
        v3 = m_yolo[key]
        if isinstance(v1, float):
            print(f"{key:<15} {v1:>12.4f} {v2:>12.4f} {v3:>12.4f}")
        else:
            print(f"{key:<15} {v1:>12} {v2:>12} {v3:>12}")

    # --- Find disagreements for visualization ---
    from src.evaluation.svm_qualitative_analysis import plot_grid

    # Cases where SVM RBF got it right but YOLO got it wrong
    svm_right_yolo_wrong = []
    # Cases where YOLO got it right but SVM RBF got it wrong
    yolo_right_svm_wrong = []

    for i, (crop_bgr, label, img_name) in enumerate(crops):
        rbf_correct = (rbf_preds[i] == label)
        yolo_correct = (yolo_preds[i] == label)

        if rbf_correct and not yolo_correct:
            svm_right_yolo_wrong.append({
                "crop": crop_bgr, "label": label,
                "img_name": img_name,
                "rbf_pred": rbf_preds[i], "yolo_pred": yolo_preds[i],
            })
        elif yolo_correct and not rbf_correct:
            yolo_right_svm_wrong.append({
                "crop": crop_bgr, "label": label,
                "img_name": img_name,
                "rbf_pred": rbf_preds[i], "yolo_pred": yolo_preds[i],
            })

    print(f"\n  SVM correct, YOLO wrong: {len(svm_right_yolo_wrong)}")
    print(f"  YOLO correct, SVM wrong: {len(yolo_right_svm_wrong)}")

    plot_grid(
        [r["crop"] for r in svm_right_yolo_wrong[:15]],
        [f"GT={'plate' if r['label'] == 1 else 'bg'}\nYOLO={'plate' if r['yolo_pred'] == 1 else 'bg'}"
         for r in svm_right_yolo_wrong[:15]],
        f"SVM Correct, YOLO Wrong ({len(svm_right_yolo_wrong)} total)",
        out / "svm_wins.png",
    )

    plot_grid(
        [r["crop"] for r in yolo_right_svm_wrong[:15]],
        [f"GT={'plate' if r['label'] == 1 else 'bg'}\nSVM={'plate' if r['rbf_pred'] == 1 else 'bg'}"
         for r in yolo_right_svm_wrong[:15]],
        f"YOLO Correct, SVM Wrong ({len(yolo_right_svm_wrong)} total)",
        out / "yolo_wins.png",
    )

    # --- Generate comparison report ---
    report_lines = [
        "# Crop-Level Comparison Report",
        "",
        "All three models evaluated on the **exact same set of crops** from the test set.",
        "This is an apples-to-apples comparison: same inputs, same labels, same evaluation.",
        "",
        f"> Images sampled: {Config.MAX_EVAL_IMAGES}",
        f"> Total crops: {len(crops)} ({int(np.sum(labels))} plates, {int(len(labels) - np.sum(labels))} background)",
        f"> Negative crops per image: {Config.NEG_PER_IMAGE}",
        "",
        "## Results",
        "",
        "| Metric | SVM Linear | SVM RBF | YOLOv8n |",
        "| --- | --- | --- | --- |",
        f"| True Positives | {m_linear['tp']} | {m_rbf['tp']} | {m_yolo['tp']} |",
        f"| False Negatives | {m_linear['fn']} | {m_rbf['fn']} | {m_yolo['fn']} |",
        f"| False Positives | {m_linear['fp']} | {m_rbf['fp']} | {m_yolo['fp']} |",
        f"| True Negatives | {m_linear['tn']} | {m_rbf['tn']} | {m_yolo['tn']} |",
        f"| **Precision** | **{m_linear['precision']:.4f}** | **{m_rbf['precision']:.4f}** | **{m_yolo['precision']:.4f}** |",
        f"| **Recall** | **{m_linear['recall']:.4f}** | **{m_rbf['recall']:.4f}** | **{m_yolo['recall']:.4f}** |",
        f"| **F1** | **{m_linear['f1']:.4f}** | **{m_rbf['f1']:.4f}** | **{m_yolo['f1']:.4f}** |",
        "",
        "## Interpretation",
        "",
        "This comparison isolates the classification ability of each model.",
        "All three receive the same pre-cropped patches, no sliding window,",
        "no multi-scale search, no localization. The only question is:",
        "given this patch, does it contain a license plate?",
        "",
        "For the SVM models, this is the native task, HOG features are extracted",
        "and the SVM classifies. For YOLO, this is an unusual setup, YOLO is",
        "designed for full-image detection, so running it on small crops tests",
        "whether its learned features can still recognize a plate in isolation.",
        "",
        "## Visual Comparison",
        "",
        "### SVM correct, YOLO wrong",
        "These crops were classified correctly by SVM but misclassified by YOLO.",
        "Most are likely small, tightly cropped patches where YOLO lacks the",
        "surrounding scene context it was trained on.",
        "",
        "![SVM wins](svm_wins.png)",
        "",
        "### YOLO correct, SVM wrong",
        "These crops were classified correctly by YOLO but misclassified by SVM.",
        "Look for cases where HOG features couldn't capture the relevant pattern -",
        "unusual plate formats, heavy blur, or textures that confuse gradient histograms.",
        "",
        "![YOLO wins](yolo_wins.png)",
        ""
    ]
    (out / "crops_comparison_report.md").write_text("\n".join(report_lines))
    print(f"\nReport saved to: {out / 'crops_comparison_report.md'}")


def eval_detection_comparison():
    """
    Full-image detection comparison: run SVM (sliding window + HOG + NMS)
    and YOLO on the same test images, match predictions to ground truth
    using IoU, and compare detection-level precision/recall/F1.

    Both models receive full images and must output bounding boxes.
    """
    from src.classical.detector import SlidingWindowDetector

    out = Path("outputs/detection_comparison")
    out.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    print("Loading SVM RBF model ...")
    clf = PlateClassification.load(Config.SVM_FULL)
    extractor = HOGFeatureExtractor(target_size=Config.HOG_TARGET_SIZE)

    print("Loading YOLO model ...")
    yolo_model = YOLO(Config.YOLO_MODEL, task="detect")

    detector = SlidingWindowDetector(
        classifier=clf,
        hog_fn=extractor.compute_single,
        window_size=(120, 40),
        step_size=16,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        score_threshold=0.5,
        iou_threshold=0.3,
    )

    # --- Collect test images ---
    img_names = sorted([
        f for f in os.listdir(Config.TEST_IMAGES)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    random.seed(Config.SEED)
    if len(img_names) > Config.MAX_EVAL_IMAGES:
        img_names = random.sample(img_names, Config.MAX_EVAL_IMAGES)

    print(f"Running detection on {len(img_names)} test images ...")

    # Counters
    svm_total_tp, svm_total_fp, svm_total_fn = 0, 0, 0
    yolo_total_tp, yolo_total_fp, yolo_total_fn = 0, 0, 0
    svm_iou_sum, yolo_iou_sum = 0.0, 0.0
    svm_perfect, yolo_perfect = 0, 0
    svm_times, yolo_times = [], []
    iou_threshold = 0.5

    # Visual examples
    svm_wins = []  # SVM found plate, YOLO missed
    yolo_wins = []  # YOLO found plate, SVM missed
    both_missed = []  # Both missed

    for i, img_name in enumerate(img_names):
        img_path = os.path.join(Config.TEST_IMAGES, img_name)
        label_path = os.path.join(Config.TEST_LABELS, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = parse_yolo_label(label_path, img_w, img_h) if os.path.exists(label_path) else []

        # --- SVM sliding window detection (timed) ---
        svm_start = time.time()
        svm_detections = detector.detect(img)
        svm_elapsed = time.time() - svm_start
        svm_times.append(svm_elapsed)

        svm_pred_boxes = [(d.x, d.y, d.x + d.w, d.y + d.h, d.score) for d in svm_detections]
        svm_tp_pairs, svm_fp, svm_fn = match_predictions(svm_pred_boxes, gt_boxes, iou_threshold)
        svm_total_tp += len(svm_tp_pairs)
        svm_total_fp += len(svm_fp)
        svm_total_fn += len(svm_fn)
        for _, _, iou_val in svm_tp_pairs:
            svm_iou_sum += iou_val

        svm_is_perfect = (len(svm_fp) == 0 and len(svm_fn) == 0 and len(svm_tp_pairs) > 0)
        if svm_is_perfect:
            svm_perfect += 1

        # --- YOLO detection (timed) ---
        yolo_start = time.time()
        yolo_results = yolo_model.predict(img, conf=0.25, verbose=False)
        yolo_elapsed = time.time() - yolo_start
        yolo_times.append(yolo_elapsed)

        yolo_pred_boxes = []
        if yolo_results and len(yolo_results[0].boxes):
            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                yolo_pred_boxes.append((x1, y1, x2, y2, conf))

        yolo_tp_pairs, yolo_fp, yolo_fn = match_predictions(yolo_pred_boxes, gt_boxes, iou_threshold)
        yolo_total_tp += len(yolo_tp_pairs)
        yolo_total_fp += len(yolo_fp)
        yolo_total_fn += len(yolo_fn)
        for _, _, iou_val in yolo_tp_pairs:
            yolo_iou_sum += iou_val

        yolo_is_perfect = (len(yolo_fp) == 0 and len(yolo_fn) == 0 and len(yolo_tp_pairs) > 0)
        if yolo_is_perfect:
            yolo_perfect += 1

        # --- Collect visual examples ---
        short_name = img_name[:30]
        svm_found = len(svm_tp_pairs) > 0
        yolo_found = len(yolo_tp_pairs) > 0
        svm_missed_any = len(svm_fn) > 0
        yolo_missed_any = len(yolo_fn) > 0

        if svm_found and yolo_missed_any and len(svm_wins) < 9:
            # SVM detected something YOLO missed
            svm_gt = [gt for _, gt, _ in svm_tp_pairs]
            svm_pred = [p for p, _, _ in svm_tp_pairs]
            vis = draw_boxes_on_image(img, tp_gt_boxes=svm_gt, tp_pred_boxes=svm_pred,
                                      fn_boxes=list(svm_fn), fp_boxes=list(svm_fp))
            svm_wins.append((vis, f"{short_name}\nSVM: {len(svm_tp_pairs)}TP {len(svm_fn)}FN"))

        if yolo_found and svm_missed_any and len(yolo_wins) < 9:
            # YOLO detected something SVM missed
            yolo_gt = [gt for _, gt, _ in yolo_tp_pairs]
            yolo_pred = [p for p, _, _ in yolo_tp_pairs]
            vis = draw_boxes_on_image(img, tp_gt_boxes=yolo_gt, tp_pred_boxes=yolo_pred,
                                      fn_boxes=list(yolo_fn), fp_boxes=list(yolo_fp))
            yolo_wins.append((vis, f"{short_name}\nYOLO: {len(yolo_tp_pairs)}TP {len(yolo_fn)}FN"))

        if svm_missed_any and yolo_missed_any and len(both_missed) < 9:
            vis = draw_boxes_on_image(img, tp_gt_boxes=gt_boxes)
            both_missed.append((vis, f"{short_name}\nBoth missed plates"))

        if (i + 1) % 25 == 0:
            svm_avg = np.mean(svm_times[-25:])
            yolo_avg = np.mean(yolo_times[-25:])
            print(f"  Processed {i + 1}/{len(img_names)} | SVM: {svm_avg:.2f}s/img | YOLO: {yolo_avg:.3f}s/img")

    # --- Compute metrics ---
    def det_metrics(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    m_svm = det_metrics(svm_total_tp, svm_total_fp, svm_total_fn)
    m_yolo = det_metrics(yolo_total_tp, yolo_total_fp, yolo_total_fn)

    svm_avg_iou = svm_iou_sum / svm_total_tp if svm_total_tp > 0 else 0
    yolo_avg_iou = yolo_iou_sum / yolo_total_tp if yolo_total_tp > 0 else 0
    svm_mean_time = np.mean(svm_times) if svm_times else 0
    yolo_mean_time = np.mean(yolo_times) if yolo_times else 0
    total_images = len(img_names)

    # --- Print ---
    print(f"\n{'=' * 65}")
    print(f"FULL-IMAGE DETECTION COMPARISON - {total_images} images @ IoU={iou_threshold}")
    print(f"{'=' * 65}")
    print(f"{'Metric':<25} {'SVM (sliding window)':>22} {'YOLOv8n':>12}")
    print(f"{'-' * 25} {'-' * 22} {'-' * 12}")
    for key in ["tp", "fp", "fn", "precision", "recall", "f1"]:
        v1 = m_svm[key]
        v2 = m_yolo[key]
        if isinstance(v1, float):
            print(f"{key:<25} {v1:>22.4f} {v2:>12.4f}")
        else:
            print(f"{key:<25} {v1:>22} {v2:>12}")
    print(f"{'avg IoU (TP)':<25} {svm_avg_iou:>22.4f} {yolo_avg_iou:>12.4f}")
    print(f"{'perfect images':<25} {svm_perfect:>22} {yolo_perfect:>12}")
    print(f"{'avg time/image':<25} {svm_mean_time:>21.3f}s {yolo_mean_time:>11.3f}s")
    print(f"{'speed ratio':<25} {svm_mean_time / yolo_mean_time if yolo_mean_time > 0 else 0:>21.0f}x {'1x':>12}")

    # --- Plot visual examples ---
    print("\nGenerating visualizations ...")
    if svm_wins:
        plot_image_grid(
            [img for img, _ in svm_wins[:9]],
            [t for _, t in svm_wins[:9]],
            f"SVM Detected, YOLO Missed",
            out / "svm_wins.png",
        )
    if yolo_wins:
        plot_image_grid(
            [img for img, _ in yolo_wins[:9]],
            [t for _, t in yolo_wins[:9]],
            f"YOLO Detected, SVM Missed",
            out / "yolo_wins.png",
        )
    if both_missed:
        plot_image_grid(
            [img for img, _ in both_missed[:9]],
            [t for _, t in both_missed[:9]],
            f"Both Models Missed",
            out / "both_missed.png",
        )

    # --- Generate report ---
    report_lines = [
        "# Full-Image Detection Comparison",
        "",
        "Both models evaluated on the **exact same test images** as full-image detectors.",
        "SVM uses sliding window + HOG + NMS. YOLO runs end-to-end detection.",
        "Predictions are matched to ground truth using IoU.",
        "",
        f"> Images: {total_images}",
        f"> IoU threshold: {iou_threshold}",
        f"> SVM model: `{Config.SVM_FULL}` (sliding window, step=16, scales=0.5-2.0)",
        f"> YOLO model: `{Config.YOLO_MODEL}`",
        "",
        "## Detection Metrics",
        "",
        "| Metric | SVM (Sliding Window) | YOLOv8n |",
        "| --- | --- | --- |",
        f"| True Positives | {m_svm['tp']} | {m_yolo['tp']} |",
        f"| False Positives | {m_svm['fp']} | {m_yolo['fp']} |",
        f"| False Negatives | {m_svm['fn']} | {m_yolo['fn']} |",
        f"| **Precision** | **{m_svm['precision']:.4f}** | **{m_yolo['precision']:.4f}** |",
        f"| **Recall** | **{m_svm['recall']:.4f}** | **{m_yolo['recall']:.4f}** |",
        f"| **F1** | **{m_svm['f1']:.4f}** | **{m_yolo['f1']:.4f}** |",
        "",
        "## Localization Quality",
        "",
        "| Metric | SVM | YOLOv8n |",
        "| --- | --- | --- |",
        f"| Average IoU (true positives) | {svm_avg_iou:.4f} | {yolo_avg_iou:.4f} |",
        f"| Perfect images (all plates found, no false alarms) | {svm_perfect}/{total_images} ({100 * svm_perfect / total_images:.1f}%) | {yolo_perfect}/{total_images} ({100 * yolo_perfect / total_images:.1f}%) |",
        "",
        "## Speed",
        "",
        "| Metric | SVM | YOLOv8n |",
        "| --- | --- | --- |",
        f"| Average time per image | {svm_mean_time:.3f}s | {yolo_mean_time:.3f}s |",
        f"| Speed ratio | {svm_mean_time / yolo_mean_time if yolo_mean_time > 0 else 0:.0f}x slower | 1x |",
        "",
        "## Interpretation",
        "",
        "This is the definitive comparison - both models performing their intended task",
        "on the same images. The SVM must scan the image with a sliding window at multiple",
        "scales, extract HOG features from each window, classify, and merge overlapping",
        "detections. YOLO processes the entire image in a single forward pass.",
        "",
        "The gap here reflects the full cost of hand-crafted features vs learned features.",
        "The sliding window must guess where to look and at what scale; YOLO learns this",
        "from data. Any scale mismatch, stride too coarse, or threshold too aggressive",
        "directly hurts SVM detection performance - problems YOLO doesn't have.",
        "",
        "## Visual Examples",
        "",
        "### SVM detected, YOLO missed",
        "Cases where the sliding window found a plate that YOLO did not detect.",
        "",
        "![SVM wins](svm_wins.png)",
        "",
        "### YOLO detected, SVM missed",
        "Cases where YOLO found a plate that the sliding window missed.",
        "",
        "![YOLO wins](yolo_wins.png)",
        "",
        "### Both models missed",
        "Cases where neither model detected the plate.",
        "These typically represent the hardest cases in the dataset.",
        "",
        "![Both missed](both_missed.png)",
    ]
    (out / "detection_comparison_report.md").write_text("\n".join(report_lines))
    print(f"\nReport saved to: {out / 'detection_comparison_report.md'}")


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
    "eval-crops":       (eval_crops_comparison, "Qualitative analysis - direct comparison on same crops"),
    "eval-detection":   (eval_detection_comparison, "Full-image detection comparison (SVM sliding window vs YOLO)"),
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