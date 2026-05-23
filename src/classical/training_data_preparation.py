"""
Prepare training data for the SVM classifier.

Reads images + YOLO-format label files, produces:
  - Positive crops (license plate regions)
  - Negative crops (random background patches that don't overlap the plate)
  - HOG features for all crops
  - Saved .npy arrays ready for SVM training

Usage:
    python -m src.data.prepare_training_data \
        --images-dir data/raw/train/images \
        --labels-dir data/raw/train/labels \
        --output-dir data/features \
        --neg-per-image 5

This produces:
    data/features/X_train.npy   — shape (n_samples, 3780)
    data/features/y_train.npy   — shape (n_samples,)  values 0 or 1
"""

import os
import argparse
import random
import numpy as np
import cv2
from pathlib import Path

from src.classical.hog_features import HOGFeatureExtractor


# ------------------------------------------------------------------
# YOLO label parsing
# ------------------------------------------------------------------

def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """
    Read a YOLO-format label file and convert to pixel bounding boxes.

    YOLO format per line: class x_center y_center width height (all normalised 0-1)

    Returns:
        List of (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)

            xmin = max(0, int((xc - w / 2) * img_w))
            ymin = max(0, int((yc - h / 2) * img_h))
            xmax = min(img_w, int((xc + w / 2) * img_w))
            ymax = min(img_h, int((yc + h / 2) * img_h))

            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))
    return boxes


# ------------------------------------------------------------------
# Negative (background) crop sampling
# ------------------------------------------------------------------

def iou_xyxy(box_a, box_b) -> float:
    """IoU between two (xmin, ymin, xmax, ymax) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def sample_negative_crops(
    img_h: int,
    img_w: int,
    gt_boxes: list[tuple[int, int, int, int]],
    num_negatives: int = 5,
    min_crop_size: int = 20,
    max_iou: float = 0.1,
    max_attempts: int = 200,
) -> list[tuple[int, int, int, int]]:
    """
    Randomly sample background crops that do NOT overlap with any GT plate box.

    We sample random rectangles with varied aspect ratios (roughly in the
    range of typical plate proportions) and reject any that overlap a
    ground-truth box above max_iou.

    Returns:
        List of (xmin, ymin, xmax, ymax) for accepted negative regions.
    """
    negatives = []
    attempts = 0

    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1

        # Random width/height — vary size and aspect ratio
        crop_w = random.randint(min_crop_size, max(min_crop_size + 1, img_w // 3))
        crop_h = random.randint(min_crop_size, max(min_crop_size + 1, img_h // 3))

        if crop_w >= img_w or crop_h >= img_h:
            continue

        x1 = random.randint(0, img_w - crop_w)
        y1 = random.randint(0, img_h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        candidate = (x1, y1, x2, y2)

        # Reject if it overlaps any plate box
        overlaps = any(iou_xyxy(candidate, gt) > max_iou for gt in gt_boxes)
        if not overlaps:
            negatives.append(candidate)

    return negatives


# ------------------------------------------------------------------
# Main data preparation
# ------------------------------------------------------------------

def prepare_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    neg_per_image: int = 5,
    seed: int = 42,
):
    """
    Walk through all images, extract positive + negative crops,
    compute HOG features, and save as .npy files.
    """
    random.seed(seed)
    np.random.seed(seed)

    extractor = HOGFeatureExtractor(target_size=(64, 128))

    all_features = []
    all_labels = []

    img_names = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Found {len(img_names)} images in {images_dir}")

    pos_count = 0
    neg_count = 0
    skipped = 0

    for i, img_name in enumerate(img_names):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(label_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = parse_yolo_label(label_path, img_w, img_h)

        if not gt_boxes:
            skipped += 1
            continue

        # --- Positive samples: crop each plate ---
        for (xmin, ymin, xmax, ymax) in gt_boxes:
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            try:
                feat = extractor.compute_single(crop)
                all_features.append(feat)
                all_labels.append(1)
                pos_count += 1
            except Exception as e:
                print(f"  Skipping positive crop in {img_name}: {e}")

        # --- Negative samples: random background crops ---
        neg_boxes = sample_negative_crops(img_h, img_w, gt_boxes, num_negatives=neg_per_image)
        for (xmin, ymin, xmax, ymax) in neg_boxes:
            crop = img[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            try:
                feat = extractor.compute_single(crop)
                all_features.append(feat)
                all_labels.append(0)
                neg_count += 1
            except Exception as e:
                print(f"  Skipping negative crop in {img_name}: {e}")

        # Progress update every 100 images
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(img_names)} images ...")

    # --- Save ---
    X = np.array(all_features)
    y = np.array(all_labels)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X)
    np.save(out / "y_train.npy", y)

    print(f"\nDone!")
    print(f"  Total samples : {len(y)}")
    print(f"  Positive (plate)     : {pos_count}")
    print(f"  Negative (background): {neg_count}")
    print(f"  Skipped images       : {skipped}")
    print(f"  Feature shape        : {X.shape}")
    print(f"  Saved to: {output_dir}/X_train.npy, y_train.npy")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare HOG features for SVM training")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Path to image directory (e.g. data/raw/train/images)",)
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Path to YOLO label directory (e.g. data/raw/train/labels)")
    parser.add_argument("--output-dir", type=str, default="data/features",
                        help="Where to save X_train.npy and y_train.npy")
    parser.add_argument("--neg-per-image", type=int, default=5,
                        help="Number of negative crops per image (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        neg_per_image=args.neg_per_image,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()