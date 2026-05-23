"""
Sliding-window detector using HOG + SVM.

This module ties together the full classical detection pipeline:
    image → candidate windows → HOG features → SVM score → NMS → final boxes

It provides:
- Multi-scale sliding window generation
- Non-maximum suppression (NMS)
- A Detector class that runs the end-to-end pipeline

Your teammate supplies the HOG extraction function; this module
consumes its output and handles everything else.
"""

import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class Detection:
    """A single detected bounding box."""
    x: int          # top-left x
    y: int          # top-left y
    w: int          # width
    h: int          # height
    score: float    # SVM decision score

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2)."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


# ----------------------------------------------------------------------
# Sliding window generation
# ----------------------------------------------------------------------

def sliding_windows(
    image_shape: tuple[int, int],
    window_size: tuple[int, int] = (120, 40),
    step_size: int = 16,
    scales: list[float] | None = None,
) -> list[tuple[int, int, int, int, float]]:
    """
    Generate candidate windows at multiple scales.

    Args:
        image_shape: (height, width) of the image.
        window_size: (w, h) base window size in pixels.
        step_size: stride in pixels.
        scales: list of scale factors. Each scale resizes the window.
                Default covers a reasonable range for license plates.

    Yields / returns:
        List of (x, y, w, h, scale) tuples.
    """
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    img_h, img_w = image_shape[:2]
    windows = []

    for scale in scales:
        win_w = int(window_size[0] * scale)
        win_h = int(window_size[1] * scale)

        if win_w > img_w or win_h > img_h:
            continue

        for y in range(0, img_h - win_h + 1, step_size):
            for x in range(0, img_w - win_w + 1, step_size):
                windows.append((x, y, win_w, win_h, scale))

    return windows


# ----------------------------------------------------------------------
# Non-maximum suppression
# ----------------------------------------------------------------------

def nms(detections: list[Detection], iou_threshold: float = 0.3) -> list[Detection]:
    """
    Greedy non-maximum suppression.

    Keeps the highest-scoring box, removes all boxes that overlap
    with it above `iou_threshold`, then repeats.

    Args:
        detections: list of Detection objects.
        iou_threshold: IoU above which a lower-scoring box is suppressed.

    Returns:
        Filtered list of Detection objects.
    """
    if len(detections) == 0:
        return []

    # Sort by score descending
    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_threshold]

    return keep


def _iou(a: Detection, b: Detection) -> float:
    """Compute intersection-over-union between two detections."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


# ----------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------

class SlidingWindowDetector:
    """
    End-to-end classical detector: sliding window + HOG + SVM + NMS.

    Usage:
        from src.classical.svm_classifier import PlateClassifier

        clf = PlateClassifier.load("models/svm_plate.joblib")

        detector = SlidingWindowDetector(
            classifier=clf,
            hog_fn=your_hog_extract_function,
        )
        detections = detector.detect(image)
    """

    def __init__(
        self,
        classifier,
        hog_fn,
        window_size: tuple[int, int] = (120, 40),
        step_size: int = 16,
        scales: list[float] | None = None,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.3,
    ):
        """
        Args:
            classifier: a PlateClassifier (or anything with a
                        `score_windows(X)` method returning an array of scores).
            hog_fn: callable(image_crop) → 1-D numpy feature vector.
                    This is supplied by the teammate handling HOG.
            window_size: (w, h) base window in pixels.
            step_size: sliding window stride.
            scales: list of scale factors for multi-scale detection.
            score_threshold: minimum SVM decision score to keep a window.
            iou_threshold: IoU threshold for NMS.
        """
        self.classifier = classifier
        self.hog_fn = hog_fn
        self.window_size = window_size
        self.step_size = step_size
        self.scales = scales
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run the full detection pipeline on a single image.

        Args:
            image: BGR image (as loaded by cv2.imread).

        Returns:
            List of Detection objects after NMS.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        windows = sliding_windows(
            gray.shape,
            window_size=self.window_size,
            step_size=self.step_size,
            scales=self.scales,
        )

        if len(windows) == 0:
            return []

        # Extract HOG features for every candidate window
        features = []
        valid_windows = []

        for (x, y, w, h, scale) in windows:
            crop = gray[y : y + h, x : x + w]
            try:
                feat = self.hog_fn(crop)
                features.append(feat)
                valid_windows.append((x, y, w, h))
            except Exception:
                # Skip windows that fail HOG extraction (e.g. too small)
                continue

        if len(features) == 0:
            return []

        X = np.array(features)
        scores = self.classifier.score_windows(X)

        # Keep windows above threshold
        raw_detections = []
        for (x, y, w, h), score in zip(valid_windows, scores):
            if score >= self.score_threshold:
                raw_detections.append(Detection(x, y, w, h, float(score)))

        # Non-maximum suppression
        return nms(raw_detections, iou_threshold=self.iou_threshold)