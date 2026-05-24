# Full-Image Detection Comparison

Both models evaluated on the **exact same test images** as full-image detectors.
SVM uses sliding window + HOG + NMS. YOLO runs end-to-end detection.
Predictions are matched to ground truth using IoU.

> Images: 200
> IoU threshold: 0.5
> SVM model: `models/svm_plate_rbf.joblib` (sliding window, step=16, scales=0.5-2.0)
> YOLO model: `models/yolo_plate.pt`

## Detection Metrics

| Metric | SVM (Sliding Window) | YOLOv8n |
| --- | --- | --- |
| True Positives | 82 | 205 |
| False Positives | 170 | 6 |
| False Negatives | 140 | 17 |
| **Precision** | **0.3254** | **0.9716** |
| **Recall** | **0.3694** | **0.9234** |
| **F1** | **0.3460** | **0.9469** |

## Localization Quality

| Metric | SVM | YOLOv8n |
| --- | --- | --- |
| Average IoU (true positives) | 0.6518 | 0.8566 |
| Perfect images (all plates found, no false alarms) | 42/200 (21.0%) | 194/200 (97.0%) |

## Speed

| Metric | SVM | YOLOv8n |
| --- | --- | --- |
| Average time per image | 143.347s | 0.018s |
| Speed ratio | 7788x slower | 1x |

## Interpretation

This is the definitive comparison — both models performing their intended task
on the same images. The SVM must scan the image with a sliding window at multiple
scales, extract HOG features from each window, classify, and merge overlapping
detections. YOLO processes the entire image in a single forward pass.

The gap here reflects the full cost of hand-crafted features vs learned features.
The sliding window must guess where to look and at what scale; YOLO learns this
from data. Any scale mismatch, stride too coarse, or threshold too aggressive
directly hurts SVM detection performance — problems YOLO doesn't have.

## Visual Examples

### SVM detected, YOLO missed
Cases where the sliding window found a plate that YOLO did not detect.

![SVM wins](svm_wins.png)

### YOLO detected, SVM missed
Cases where YOLO found a plate that the sliding window missed.

![YOLO wins](yolo_wins.png)

### Both models missed
Cases where neither model detected the plate.
These typically represent the hardest cases in the dataset.

![Both missed](both_missed.png)