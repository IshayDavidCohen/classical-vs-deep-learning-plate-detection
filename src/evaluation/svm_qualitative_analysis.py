"""
Qualitative analysis of SVM classifier on image crops.

Provides functions to evaluate a trained SVM model by classifying
plate and background crops from test images, then generating
visual grids of true positives, false negatives, and false positives.

Functions are called from main.py — this module has no CLI entry point.
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pylab as plt

from src.common.utils import parse_yolo_label, sample_negative_crops


def collect_predictions(images_dir, labels_dir, clf, extractor, max_images=200, neg_per_image=5, seed=42):
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
            except Exception as e:
                print(f"  Skipping crop in {img_name}: {e}")
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
            except Exception as e:
                print(f"  Skipping crop in {img_name}: {e}")
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
