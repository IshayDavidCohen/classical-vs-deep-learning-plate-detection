# Classical vs Deep Learning for License Plate Detection

A comparison of **HOG + SVM** (classical computer vision) against **YOLOv8n** (deep learning) for license plate detection.

This is a CS machine learning final project. We don't read the plate text - we only detect **where the plate is** in the image.

## Research Question

> How far can a classical detection pipeline go compared to a modern deep learning detector, and under what conditions does each approach succeed or fail?

## Results at a Glance

| Metric | HOG+SVM (Linear) | HOG+SVM (RBF) | YOLOv8n |
| --- | --- | --- | --- |
| Precision | 0.9484 | 0.99 | 0.9919 |
| Recall | 0.9606 | 0.98 | 0.9475 |
| F1 | 0.9544 | 0.98 | ~0.969 |
| mAP50 | - | - | 0.9741 |
| Training time | ~13 min | ~5 hours | 76 min |

**Note:** SVM metrics are crop-level classification. YOLO metrics are full-image detection with IoU matching. See the [report](./outputs/model_comparison_report.md) for a detailed comparison.

## Dataset

[License Plate Detection Dataset](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images) - 10,125 annotated images with YOLO-format bounding box labels.

| Split | Images |
| --- | --- |
| Train | ~7,057 |
| Valid | ~2,048 |
| Test | ~1,020 |

Download the dataset and place it in `data/raw/` with `train/`, `valid/`, and `test/` subdirectories, each containing `images/` and `labels/`.

## Setup

```bash
# Clone
git clone https://github.com/<your-repo>/classical-vs-deep-learning-plate-detection.git
cd classical-vs-deep-learning-plate-detection

# Install dependencies
pip install -r requirements.txt
```

For GPU support (YOLO training):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Pre-trained Models

Download from the [v0.9-beta-trained-models](https://github.com/IshayDavidCohen/classical-vs-deep-learning-plate-detection/releases/tag/v-0.9-beta-trained-models) release and place in `models/`:

```
models/
├── svm_plate_linear.joblib
├── svm_plate_rbf.joblib
└── yolo_plate.pt
```

## Usage

Everything runs from `main.py`:

```bash
python main.py help
```

```
Available commands:

  prepare-data         Extract HOG features from training images
  train-svm-linear     Train SVM - linear kernel only
  train-svm-full       Train SVM - full grid (linear + RBF)
  train-yolo           Train YOLOv8n detector
  eval-svm-linear      Qualitative analysis - SVM linear
  eval-svm-full        Qualitative analysis - SVM full (best)
  eval-yolo            Qualitative analysis - YOLO
  eval-all             Run all three qualitative analyses
```

### Train from scratch

```bash
# 1. Extract HOG features (positive plates + negative background crops)
python main.py prepare-data

# 2. Train SVM (linear only - fast, ~13 min)
python main.py train-svm-linear

# 3. Train SVM (full grid search - linear + RBF, ~5 hours)
python main.py train-svm-full

# 4. Train YOLOv8n (~76 min on RTX 5070 Ti)
python main.py train-yolo
```

### Evaluate existing models

```bash
# Run qualitative analysis on all three models
python main.py eval-all

# Or individually
python main.py eval-svm-linear
python main.py eval-svm-full
python main.py eval-yolo
```

Each evaluation generates a markdown report with visual grids (TP/FN/FP) in `outputs/`.

## How It Works

### Classical Pipeline: HOG + SVM

```
image → candidate windows → HOG features → SVM score → NMS → bounding boxes
```

1. **Training data preparation** - for each training image, crop the plate region (positive) and sample random background patches (negative). Extract HOG features from each crop.
2. **SVM training** - train a binary classifier (plate vs background) on the HOG feature vectors using grid search over kernel and regularization parameters.
3. **Detection** - at inference, slide windows across the image at multiple scales, extract HOG from each window, score with SVM, and apply non-maximum suppression.

HOG parameters: 64×128 target size, 9 orientations, 8×8 pixels per cell, 2×2 cells per block → 3,780-dimensional feature vector.

### Deep Learning Pipeline: YOLOv8n

```
image → neural network → bounding boxes + confidence
```

Fine-tune a pretrained YOLOv8n (nano) on the same dataset. YOLO handles feature extraction, classification, and localization in a single forward pass.

## Project Structure

```
├── main.py                              # CLI entry point - runs everything
├── paper/
│   └── report.md                        # Project report
├── src/
│   ├── classical/
│   │   ├── hog_features.py              # HOG feature extraction
│   │   ├── svm_classifier.py            # SVM wrapper (train, predict, save/load)
│   │   ├── detector.py                  # Sliding window + NMS
│   │   ├── train_svm.py                 # Training logic + report generation
│   │   └── training_data_preparation.py # Positive + negative crop sampling
│   ├── deep/
│   │   └── train_yolo.py                # YOLO training + report generation
│   ├── evaluation/
│   │   ├── metrics.py                   # Classification + detection metrics
│   │   ├── svm_qualitative_analysis.py  # Visual analysis of SVM predictions
│   │   └── yolo_qualitative_analysis.py # Visual analysis of YOLO predictions
│   └── common/
│       └── utils.py                     # Shared utilities (YOLO label parsing, IoU, etc.)
├── models/                              # Trained model files (.joblib, .pt)
├── outputs/                             # Generated reports, plots, confusion matrices
└── data/
    ├── features/                        # HOG feature vectors (.npy)
    └── raw/                             # Dataset (train/valid/test splits)
```

## Key Findings

**The classical pipeline performs surprisingly well on crop classification.** The RBF SVM achieves 98% F1 on classifying pre-cropped patches - comparable to YOLOv8n's precision and recall on full-image detection.

**The real gap appears in full-image detection.** YOLO handles multi-scale detection, localization, and classification in a single pass. The classical pipeline requires sliding windows, manual scale selection, and NMS - each introducing potential failure points.

**Failure patterns differ between approaches:**
- **SVM fails on:** severely blurry crops, unusual plate formats (e.g. Dubai plates), extreme rotation, and text-like background objects (signs, banners)
- **YOLO fails on:** extremely dense multi-plate scenes (traffic jams with 20+ tiny plates) and unannotated plates in the ground truth

**Practical tradeoff:** Linear SVM trains in 13 minutes on CPU. RBF takes 5 hours for a 2.5% F1 improvement. YOLO takes 76 minutes on GPU but delivers end-to-end detection without feature engineering.

## References

- Dalal, N. and Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- [License Plate Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images)