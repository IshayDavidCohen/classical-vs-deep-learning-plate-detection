"""
Qualitative analysis of the YOLOv8n detector on test images.

Loads the trained YOLO model, runs detection on test images,
compares predictions to ground-truth labels, and visualizes:
  - True Positives  (correctly detected plates with IoU ≥ threshold)
  - False Negatives (plates the model missed entirely)
  - False Positives (spurious detections with no matching GT)

Saves annotated image grids to the output directory for the report.

Usage:
    python -m src.evaluation.qualitative_analysis_yolo \
        --images-dir data/raw/test/images \
        --labels-dir data/raw/test/labels \
        --model-path models/yolo_plate.pt \
        --output-dir outputs/qualitative_yolo
"""

import os
import random
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

from src.common.utils import parse_yolo_label


def compute_iou_xyxy(box_a, box_b) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to GT boxes greedily by IoU.

    Args:
        pred_boxes: list of (x1, y1, x2, y2, conf)
        gt_boxes:   list of (x1, y1, x2, y2)
        iou_threshold: minimum IoU for a match

    Returns:
        tp_pairs: list of (pred_box, gt_box, iou) — matched
        fp_boxes: list of pred_box — unmatched predictions
        fn_boxes: list of gt_box — unmatched ground truths
    """
    # Sort preds by confidence descending
    preds_sorted = sorted(pred_boxes, key=lambda b: b[4], reverse=True)
    matched_gt = set()

    tp_pairs = []
    fp_boxes = []

    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou_xyxy(pred[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_pairs.append((pred, gt_boxes[best_gt_idx], best_iou))
            matched_gt.add(best_gt_idx)
        else:
            fp_boxes.append(pred)

    fn_boxes = [gt for i, gt in enumerate(gt_boxes) if i not in matched_gt]
    return tp_pairs, fp_boxes, fn_boxes


# def draw_boxes_on_image(img, gt_boxes=None, pred_boxes=None, title=""):
#     """
#     Draw GT (green) and predicted (red) boxes on an image.
#     Returns the annotated image in RGB.
#     """
#     vis = img.copy()
#     if gt_boxes:
#         for box in gt_boxes:
#             x1, y1, x2, y2 = [int(v) for v in box[:4]]
#             cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     if pred_boxes:
#         for box in pred_boxes:
#             x1, y1, x2, y2 = [int(v) for v in box[:4]]
#             conf = box[4] if len(box) > 4 else 0
#             cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#     return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def draw_boxes_on_image(img, tp_gt_boxes=None, fn_boxes=None, fp_boxes=None, tp_pred_boxes=None):
    """
    Draw all detection outcomes on a single image with a legend.
    Green  = matched GT (true positive)
    Yellow = missed GT (false negative)
    Red    = unmatched prediction (false positive)
    Blue   = matched prediction (true positive)
    """
    vis = img.copy()

    # Draw legend bar at the top
    img_w = vis.shape[1]
    legend_h = max(30, int(img_w * 0.04))
    font_scale = max(0.35, img_w / 2000)
    box_size = max(12, int(legend_h * 0.5))
    padding = int(legend_h * 0.25)

    legend = np.zeros((legend_h, img_w, 3), dtype=np.uint8)
    legend[:] = (40, 40, 40)

    items = [
        ((0, 255, 0), "GT matched"),
        ((255, 0, 0), "Pred matched"),
        ((0, 255, 255), "GT missed"),
        ((0, 0, 255), "False det"),
    ]
    spacing = img_w // (len(items) + 1)
    for idx, (color, text) in enumerate(items):
        x = 10 + idx * spacing
        cv2.rectangle(legend, (x, padding), (x + box_size, padding + box_size), color, -1)
        cv2.putText(legend, text, (x + box_size + 5, padding + box_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    vis = np.vstack([legend, vis])

    # Draw boxes (same as before)
    offset = legend_h  # shift y coords down by legend height
    if tp_gt_boxes:
        for box in tp_gt_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(vis, (x1, y1 + offset), (x2, y2 + offset), (0, 255, 0), 2)
    if tp_pred_boxes:
        for box in tp_pred_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            conf = box[4] if len(box) > 4 else 0
            cv2.rectangle(vis, (x1, y1 + offset), (x2, y2 + offset), (255, 0, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1 + offset - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if fn_boxes:
        for box in fn_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(vis, (x1, y1 + offset), (x2, y2 + offset), (0, 255, 255), 2)
    if fp_boxes:
        for box in fp_boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            conf = box[4] if len(box) > 4 else 0
            cv2.rectangle(vis, (x1, y1 + offset), (x2, y2 + offset), (0, 0, 255), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1 + offset - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def plot_image_grid(images, titles, suptitle, save_path, cols=3, max_items=9):
    """Plot a grid of annotated images."""
    items = images[:max_items]
    item_titles = titles[:max_items]

    if len(items) == 0:
        print(f"  No items to plot for: {suptitle}")
        return

    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(items, item_titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    for j in range(len(items), len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
