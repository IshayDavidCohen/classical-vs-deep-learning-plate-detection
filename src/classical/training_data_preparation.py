"""
Prepare training data for the SVM classifier.

Reads images + YOLO-format label files, produces:
  - Positive crops (license plate regions)
  - Negative crops (random background patches that don't overlap the plate)
  - HOG features for all crops
  - Saved .npy arrays ready for SVM training

Usage:
    python -m src.data.prepare_training_data \
        --images-dir G:/coding/repos/classical-vs-deep-learning-plate-detection/data/raw/train/images
        --labels-dir G:/coding/repos/classical-vs-deep-learning-plate-detection/data/raw/train/labels
        --output-dir G:/coding/repos/classical-vs-deep-learning-plate-detection/data/features
        --neg-per-image 5

This produces:
    data/features/X_train.npy   - shape (n_samples, 3780)
    data/features/y_train.npy   - shape (n_samples,)  values 0 or 1
"""

import os
import argparse
import random
import numpy as np
import cv2
from pathlib import Path

from src.classical.hog_features import HOGFeatureExtractor
from src.common.utils import parse_yolo_label, sample_negative_crops


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
