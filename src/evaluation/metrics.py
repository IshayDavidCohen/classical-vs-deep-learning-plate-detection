"""
Evaluation utils for both classification and detection

Two levels of eval:
1. Classification: How good well does the SVM classify an individual crop of an image
2. Detection: How well does the full pipeline localize plates?

per image IoU matching and AP-style metrics.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ======================================================================
# 1. Classification-level metrics (SVM crop evaluation)
# ======================================================================

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    print_report: bool = True,
) -> dict:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: ground-truth labels (0/1).
        y_pred: predicted labels (0/1).
        print_report: whether to print sklearn's classification report.

    Returns:
        dict with precision, recall, f1, and confusion_matrix.
    """
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if print_report:
        print(classification_report(y_true, y_pred, target_names=["background", "plate"]))

    return {"precision": p, "recall": r, "f1": f1, "confusion_matrix": cm}


# ======================================================================
# 2. Detection-level metrics (bounding-box evaluation)
# ======================================================================

def compute_iou(box_a, box_b) -> float:
    """
    IoU between two boxes in (x, y, w, h) format.
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def match_detections(
    pred_boxes: list[tuple],
    gt_boxes: list[tuple],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """
    Greedy matching of predicted boxes to ground-truth boxes.

    Each GT box can be matched at most once. Predictions are
    sorted by score (if 5-tuples with score as last element)
    or processed in order.

    Args:
        pred_boxes: list of (x, y, w, h) or (x, y, w, h, score).
        gt_boxes:   list of (x, y, w, h).
        iou_threshold: minimum IoU to count as a match.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    # Sort predictions by score descending if scores are available
    if pred_boxes and len(pred_boxes[0]) == 5:
        pred_boxes = sorted(pred_boxes, key=lambda b: b[4], reverse=True)

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        pred_box = pred[:4]
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def detection_metrics(
    all_pred_boxes: list[list[tuple]],
    all_gt_boxes: list[list[tuple]],
    iou_threshold: float = 0.5,
    print_summary: bool = True,
) -> dict:
    """
    Aggregate detection metrics over a dataset.

    Args:
        all_pred_boxes: list of per-image prediction lists.
                        Each prediction is (x, y, w, h) or (x, y, w, h, score).
        all_gt_boxes:   list of per-image ground-truth lists.
                        Each GT box is (x, y, w, h).
        iou_threshold:  IoU threshold for matching.
        print_summary:  whether to print results.

    Returns:
        dict with precision, recall, f1, total_tp, total_fp, total_fn.
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    for preds, gts in zip(all_pred_boxes, all_gt_boxes):
        tp, fp, fn = match_detections(preds, gts, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }

    if print_summary:
        print(f"Detection metrics @ IoU={iou_threshold:.2f}")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")

    return results