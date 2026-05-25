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

## What we actually tested
The HOG + SVM pipeline dealt with the full detection problem: given a raw image, can it find and localize the plate?
The sliding window detector scanned each full image by moving a window across it at 6 different scales, step by step (every 16px)
For a typical 1024x768 image, that's roughly 15,000-30,000 windows. Each window was cropped, passed through HOG, and scored by the SVM.
High-scoring windows were kept, then NMS merged overlapping ones into a final bounding boxes. 
Meanwhile, YOLOv8n processed the entire image in a single forward pass.

Any scale mismatch, stride too coarse, or threshold too aggressive
directly hurts SVM detection performance, problems our YOLO doesn't have.

## Interpretation

The SVM went from 97.5% F1 on crops to 34.6% F1 on full images. That's not a minor degradation, the pipeline essentially collapsed under the weight of the detection task.


170 false positives means the sliding window found plate-like patterns in bumper stickers, signs, text on buildings, and vehicle grilles. 140 false negatives means it missed most actual plates because the window never landed at the right scale or position. Only 42 out of 200 images were handled without error.
YOLO held steady at 94.7% F1 with 97% of images perfect, processing each one in 18 milliseconds.

This exposes the core limitation of the classical pipeline. The SVM's 98% crop accuracy was real, it genuinely knows what a plate looks like. But detection is not classification. The sliding window introduces three problems that the crop evaluation hid:

1. **Scale mismatch** - the window sizes don't always match the actual plate size in the image, so some plates are never seen at the right scale
2. **Localization noise** - the window rarely lands perfectly centered on a plate, producing partial overlaps that fail the IoU threshold
3. **False positive explosion** - scanning 15,000+ windows per image means 15,000 chances to be wrong, and even a 1% false positive rate on crops becomes hundreds of spurious detections per image

YOLO sidesteps all three because it learned localization, scale handling, and classification jointly from data.

One additional finding: there were zero cases where SVM detected a plate that YOLO missed. YOLO's detection coverage fully contains the SVM's, everything the classical pipeline found, YOLO also found, plus much more.

## The Full Picture

| Setting | SVM F1 | YOLO F1 | Winner |
| --- | --- | --- | --- |
| Crop classification (SVM's home turf) | 0.9748 | 0.8102 | SVM |
| Full-image detection (YOLO's home turf) | 0.3460 | 0.9469 | YOLO |

Classical methods can match or beat deep learning when the problem is well-scoped - give the SVM a cropped patch, and it classifies better than YOLO. But real-world detection isn't well-scoped. The plate could be anywhere, at any scale, in any context. That's where end-to-end learning earns its advantage.


## Visual Examples

### SVM detected, YOLO missed
There were no cases where the sliding window found a plate that YOLO did not detect.
In other words it is safe to say, YOLOv8 nano, aced this test compared to the classical approach.

### YOLO detected, SVM missed
Cases where YOLO found a plate that the sliding window missed.

![YOLO wins](yolo_wins.png)

### Both models missed
Cases where neither model detected the plate.
These typically represent the hardest cases in the dataset.

![Both missed](both_missed.png)