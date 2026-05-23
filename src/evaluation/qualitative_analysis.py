"""
Qualitative analysis of SVM classifier on image crops
This acts as our evaluation file for the results of our .joblib model

We will load the trained SVM mode, extract plate + background crops the dataset.
split, classify them, and will visualize:
    - True Positives (TP): correctly classified plate crop
    - False Positives (FP): background crop incorrectly classified as plate
    - True Negatives (TN): correctly classified background crop
Obviously it wont have any FN since we are only evaluating plates.

It saves visual grids to the output directory for use in the report.md

Usage:
    python -m src.evaluation.qualitative_analysis \
        --images-dir data/raw/test/images \
        --labels-dir data/raw/test/labels \
        --model-path models/svm_plate.joblib \
        --output-dir outputs/qualitative

"""

import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pylab as plt

from pathlib import Path

from src.classical.hog_features import HOGFeatureExtractor
from src.classical.svm_classifier import PlateClassification
from src.common.utils import parse_yolo_label, sample_negative_crops


def collect_predictions(images_dir, labels_dir, clf, extractor, max_images=200, neg_per_image=2, seed=42):
    """
    Run the classifier on plate crops + background crops from images.

    Returns list of dicts with: crop, label, pred, score, img_name, box
    """
    random.seed(seed)
    results = []

    img_names = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if max_images and len(img_names) > max_images:
        img_names = random.sample(img_names, max_images)

    for img_name in img_names:
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

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
        for box in gt_boxes:
            xmin, ymin, xmax, ymax = box
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            try:
                feat = extractor.compute_single(crop)
                pred, score = clf.predict(feat.reshape(1, -1))
                results.append({
                    "crop": crop,
                    "label": 1,
                    "pred": int(pred[0]),
                    "score": float(score[0]),
                    "img_name": img_name,
                    "box": box,
                })
            except Exception:
                continue

        # Negative crops
        neg_boxes = sample_negative_crops(img_h, img_w, gt_boxes, num_negatives=neg_per_image)
        for neg_box in neg_boxes:
            xmin, ymin, xmax, ymax = neg_box
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            try:
                feat = extractor.compute_single(crop)
                pred, score = clf.predict(feat.reshape(1, -1))
                results.append({
                    "crop": crop,
                    "label": 0,
                    "pred": int(pred[0]),
                    "score": float(score[0]),
                    "img_name": img_name,
                    "box": neg_box,
                })
            except Exception:
                continue

    return results


def plot_grid(crops, titles, suptitle, save_path, cols=5, max_items=15):
    """Plot a grid of crops with titles."""
    items = crops[:max_items]
    item_titles = titles[:max_items]

    if len(items) == 0:
        print(f"  No items to plot for: {suptitle}")
        return

    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (crop, title) in enumerate(zip(items, item_titles)):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        axes[i].imshow(crop_rgb)
        axes[i].set_title(title, fontsize=8)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(items), len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Qualitative analysis of SVM classifier")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Path to images (e.g. data/raw/test/images)")
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Path to YOLO labels (e.g. data/raw/test/labels)")
    parser.add_argument("--model-path", type=str, default="models/svm_plate.joblib",
                        help="Path to trained SVM model")
    parser.add_argument("--output-dir", type=str, default="outputs/qualitative",
                        help="Where to save analysis images")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Max images to analyze (for speed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model and extractor
    print("Loading model ...")
    clf = PlateClassification.load(args.model_path)
    extractor = HOGFeatureExtractor(target_size=(64, 128))

    # Collect predictions
    print(f"Running classifier on crops from {args.images_dir} ...")
    results = collect_predictions(
        args.images_dir, args.labels_dir, clf, extractor,
        max_images=args.max_images, seed=args.seed,
    )

    # Categorize
    tp = [r for r in results if r["label"] == 1 and r["pred"] == 1]
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]
    tn = [r for r in results if r["label"] == 0 and r["pred"] == 0]

    print(f"\nResults on {len(results)} crops:")
    print(f"  True Positives:  {len(tp)}")
    print(f"  False Negatives: {len(fn)}")
    print(f"  False Positives: {len(fp)}")
    print(f"  True Negatives:  {len(tn)}")

    # Sort by score for interesting examples
    tp_sorted = sorted(tp, key=lambda r: r["score"])  # lowest-confidence TPs
    fn_sorted = sorted(fn, key=lambda r: r["score"])  # most-missed plates
    fp_sorted = sorted(fp, key=lambda r: r["score"], reverse=True)  # highest-confidence FPs

    # Plot grids
    print("\nGenerating visualizations ...")

    plot_grid(
        [r["crop"] for r in tp_sorted[:15]],
        [f"score={r['score']:.2f}" for r in tp_sorted[:15]],
        f"True Positives — Lowest Confidence ({len(tp)} total)",
        out / "true_positives_low_confidence.png",
    )

    plot_grid(
        [r["crop"] for r in fn_sorted[:15]],
        [f"score={r['score']:.2f}\n{r['img_name'][:25]}" for r in fn_sorted[:15]],
        f"False Negatives — Missed Plates ({len(fn)} total)",
        out / "false_negatives.png",
    )

    plot_grid(
        [r["crop"] for r in fp_sorted[:15]],
        [f"score={r['score']:.2f}\n{r['img_name'][:25]}" for r in fp_sorted[:15]],
        f"False Positives — Background Called Plate ({len(fp)} total)",
        out / "false_positives.png",
    )

    # --- Summary report ---
    total_pos = len(tp) + len(fn)
    total_neg = len(fp) + len(tn)
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary_lines = [
        "# Qualitative Analysis Report",
        "",
        f"> Dataset: `{args.images_dir}`",
        f"> Model: `{args.model_path}`",
        f"> Images analyzed: {args.max_images}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total crops | {len(results)} |",
        f"| Positive crops (plates) | {total_pos} |",
        f"| Negative crops (background) | {total_neg} |",
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
        "They show the boundary of what the model considers a plate.",
        "",
        "![True Positives](true_positives_low_confidence.png)",
        "",
        "### False Negatives (missed plates)",
        "These are actual plates that the model classified as background.",
        "Look for patterns: small plates, blur, occlusion, unusual angles.",
        "",
        "![False Negatives](false_negatives.png)",
        "",
        "### False Positives (background called plate)",
        "These are background patches the model mistakenly called plates.",
        "Look for patterns: rectangular shapes, text-like textures, high contrast edges.",
        "",
        "![False Positives](false_positives.png)",
        "",
    ]

    report_path = out / "qualitative_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nReport saved to: {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
