# Model Comparison Report

> Classical (HOG + SVM) vs Deep Learning (YOLOv8n) for License Plate Detection

## Training Overview

| Parameter | HOG + SVM (Linear) | HOG + SVM (RBF) | YOLOv8n |
| --- | --- | --- | --- |
| Model type | StandardScaler + SVC | StandardScaler + SVC | YOLOv8n (nano) |
| Feature extraction | HOG (3,780-dim) | HOG (3,780-dim) | Learned (end-to-end) |
| Training samples | 34,093 crops | 34,093 crops | 7,057 images |
| Validation samples | 8,524 crops | 8,524 crops | 2,048 images |
| Test samples | - | - | 1,020 images |
| Best hyperparams | C=0.1, linear | C=10, rbf, gamma=scale | batch=16, 100 epochs |
| Training time | ~13 min | ~5 hours | 76 min |
| Hardware | Intel i9-10800K (CPU) | Intel i9-10800K (CPU) | RTX 5070 Ti (GPU) |

## Classification / Detection Metrics

| Metric | HOG + SVM (Linear) | HOG + SVM (RBF) | YOLOv8n |
| --- | --- | --- | --- |
| Precision | 0.9484 | 0.99 | **0.9919** |
| Recall | 0.9606 | 0.98 | 0.9475 |
| F1 Score | 0.9544 | 0.98 | ~0.969 |
| mAP50 | - | - | 0.9741 |
| mAP50-95 | - | - | 0.7241 |

## Confusion Matrix (SVM Linear - Validation)

| | Predicted Background | Predicted Plate |
| --- | --- | --- |
| **Actual Background** | 6,975 (TN) | 77 (FP) |
| **Actual Plate** | 58 (FN) | 1,414 (TP) |

## Qualitative Analysis (SVM Linear - Test Set, 200 images)

| Metric | Value |
| --- | --- |
| Total crops evaluated | 620 |
| True Positives | 206 |
| False Negatives | 16 |
| False Positives | 2 |
| True Negatives | 396 |
| Precision | 0.9904 |
| Recall | 0.9279 |
| F1 | 0.9581 |

## Key Observations

### SVM matches YOLO on crop-level precision

On isolated patch classification, the RBF SVM achieves precision and recall comparable to YOLOv8n. This shows that HOG features combined with a well-tuned SVM can effectively distinguish license plate texture from background, the hand-crafted gradient features capture enough plate-specific structure to perform well in controlled conditions.

### The metrics are not directly comparable (yet)

The SVM numbers reflect **crop classification** - given a pre-cropped patch, is it a plate or not? The YOLO numbers reflect **full-image detection** - given a complete image, can the model locate and box the plate with sufficient IoU? These are fundamentally different tasks. The SVM has not yet been evaluated as a full-image detector (via sliding window + NMS), so the comparison is partial.

### Linear vs RBF tradeoff

The RBF kernel improves F1 from 0.9544 to 0.98 (+2.6 percentage points) but takes roughly 23× longer to train. For a student project, linear is practical for iteration; RBF is worth running once for the best final numbers.

### SVM failure modes (from qualitative analysis)

The classical pipeline fails on:

- **Severely blurry or low-resolution crops** - HOG features become meaningless when gradients are smeared
- **Unusual plate formats** - plates with non-standard layouts (Dubai, vertical, decorative) that differ from the training distribution
- **Extreme rotation** - HOG is not rotation-invariant, so tilted plates produce unfamiliar gradient patterns
- **Text-like background objects** - signs and banners with rectangular shapes and strong edges trigger false positives (e.g., "LAND OF LINCOLN" banner)

### What remains for a fair comparison

To complete the comparison, the SVM needs to be evaluated as a full-image detector using the sliding window pipeline. This will likely show a significant drop in SVM performance relative to YOLO, because the sliding window introduces challenges (scale mismatch, window positioning, many more false positive opportunities) that YOLO handles natively through its learned multi-scale detection heads.