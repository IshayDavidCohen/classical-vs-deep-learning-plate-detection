# License Plate Detection: HOG+SVM vs YOLOv8n

A comparative study of classical (HOG + SVM) and deep learning (YOLOv8n) approaches for license plate detection.

**Authors:** Ishay Cohen, Mikhael Pelagein

> How far can a classical detection pipeline go compared to modern deep learning, and under what conditions does each approach succeed or fail?

## Results at a Glance

| Setting | SVM RBF F1 | YOLO F1 | Winner |
| --- | --- | --- | --- |
| Crop classification | 0.9748 | 0.8102 | SVM |
| Full-image detection | 0.3460 | 0.9469 | YOLO |

Each model wins on its own turf. Full analysis in the [comparison report](paper/model_comparison_report.md).

## Dataset

[License Plate Detection Dataset](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images) - 10,125 annotated images with YOLO-format bounding box labels.

| Split | Images |
| --- | --- |
| Train | ~7,057 |
| Valid | ~2,048 |
| Test | ~1,020 |

Download and extract into `data/raw/`:

```
data/raw/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

## Setup

```bash
git clone https://github.com/IshayDavidCohen/classical-vs-deep-learning-plate-detection.git
cd classical-vs-deep-learning-plate-detection
pip install -r requirements.txt
```

### Requirements

```
joblib==1.5.3
numpy==2.4.6
scikit-learn
scikit-image
opencv-python
matplotlib
ultralytics
torch
torchvision
```

For GPU support (YOLO training):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For Manim visualizations (optional, not required as it was used only for the animations in the 5 minute video):
```bash
pip install manim
```

### Pre-trained Models (Required)

Download all assets from the [v1.0-trained-models](https://github.com/IshayDavidCohen/classical-vs-deep-learning-plate-detection/releases/tag/v1.0.0-trained-models) release and place in `models/`:

```
models/
├── svm_plate_linear.joblib    # HOG+SVM, linear kernel, C=0.1 (F1=0.9544)
├── svm_plate_rbf.joblib       # HOG+SVM, RBF kernel, C=10 (F1=0.98)
└── yolo_plate.pt              # YOLOv8n, 100 epochs (mAP50=0.9741)
```

The release also includes precomputed HOG features (`X_train.npy`, `y_train.npy`) - place these in `data/features/` if you want to skip the feature extraction step (`prepare-data` function, i.e HOG phase).

## Usage

Everything runs from `main.py`:

```bash
python main.py help
```

### Available Commands

```
Data preparation:
  prepare-data         Extract HOG features from training images

Training:
  train-svm-linear     Train SVM - linear kernel only (~13 min CPU)
  train-svm-full       Train SVM - full grid search, linear + RBF (~5 hours CPU)
  train-yolo           Train YOLOv8n detector (~76 min GPU)

Evaluation:
  eval-svm-linear      Qualitative analysis - SVM linear
  eval-svm-full        Qualitative analysis - SVM full (best)
  eval-yolo            Qualitative analysis - YOLO
  eval-crops           Apples-to-apples crop comparison (all 3 models)
  eval-detection       Full-image detection comparison (SVM sliding window vs YOLO)
  eval-all             Run all three qualitative analyses
```

### Train from scratch

```bash
# 1. Extract HOG features (positive plates + negative background crops)
python main.py prepare-data

# 2. Train SVM - linear only (fast iteration)
python main.py train-svm-linear

# 3. Train SVM - full grid search (linear + RBF, finds best kernel)
python main.py train-svm-full

# 4. Train YOLOv8n (requires GPU)
python main.py train-yolo
```

### Evaluate with pre-trained models

```bash
# Individual qualitative analyses (generates TP/FN/FP grids + reports)
python main.py eval-svm-linear
python main.py eval-svm-full
python main.py eval-yolo

# Apples-to-apples crop comparison (all 3 models on same 1,217 crops)
python main.py eval-crops

# Full-image detection comparison (SVM sliding window vs YOLO, can take up to 7-8 hours)
python main.py eval-detection

# Run all three qualitative analyses at once
python main.py eval-all
```

All evaluation outputs go to `outputs/` as markdown reports with embedded images.

## How It Works

### Classical Pipeline

```
Image → Sliding Window → HOG Features → SVM Classification → NMS → Bounding Boxes
```

Each stage is hand-designed: HOG extracts a 3,780-dim gradient feature vector, SVM classifies it as plate or background, sliding window scans the image at 6 scales, and NMS merges overlapping detections.

### Deep Learning Pipeline

```
Image → YOLOv8n → Bounding Boxes + Confidence
```

A single neural network handles feature extraction, localization, and classification in one forward pass. Fine-tuned from pretrained COCO weights.

## Project Structure

```
├── main.py                              # CLI entry point - all commands
├── src/
│   ├── classical/
│   │   ├── hog_features.py              # HOG feature extraction
│   │   ├── svm_classifier.py            # SVM wrapper (train, predict, save/load)
│   │   ├── detector.py                  # Sliding window + NMS
│   │   ├── train_svm.py                 # Training utilities + report generation
│   │   └── training_data_preparation.py # Positive + negative crop sampling
│   ├── deep/
│   │   ├── train_yolo.py                # YOLO training config + report generation
│   │   └── yolov8n.pt                   # Base YOLOv8n weights (pretrained COCO)
│   ├── evaluation/
│   │   ├── metrics.py                   # Classification + detection metrics
│   │   ├── svm_qualitative_analysis.py  # Visual analysis of SVM predictions
│   │   └── yolo_qualitative_analysis.py # Visual analysis of YOLO predictions
│   └── common/
│       └── utils.py                     # YOLO label parsing, IoU, crop sampling
├── models/                              # Trained models (download from release)
├── outputs/                             # Generated reports, plots, comparisons
│   ├── model_comparison_report.md       # Full comparison report with all findings
│   ├── crops_comparison/                # Crop-level comparison results
│   ├── detection_comparison/            # Full-image detection comparison results
│   ├── qualitative_linear/              # SVM linear qualitative analysis
│   ├── qualitative_full/               # SVM RBF qualitative analysis
│   └── qualitative_yolo/               # YOLO qualitative analysis
├── animations/                          # Manim visualization scripts
│   ├── svm_visualization.py             # 2D: Linear vs RBF decision boundaries
│   └── svm_visualization_3d.py          # 3D: rotating PCA projection
├── data/                                # Dataset (not in repo, download from Kaggle)
├── legacy_output/                       # Original training outputs and logs
└── requirements.txt
```

## Key Findings

Classical ML is not obsolete - it is specialized. The SVM classifies pre-cropped patches better than YOLO (97.5% vs 81% F1). But real-world detection requires localization, and that's where the sliding window collapses (34.6% F1) while YOLO thrives (94.7% F1, 7,788x faster, zero cases where SVM found a plate YOLO missed).

## References

- Dalal, N. and Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- [License Plate Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/barkataliarbab/license-plate-detection-dataset-10125-images)

## AI Disclosure

Claude (Anthropic) was used for code assistance, debugging, and Manim visualizations. All architectural decisions, analysis, and conclusions are our own.
