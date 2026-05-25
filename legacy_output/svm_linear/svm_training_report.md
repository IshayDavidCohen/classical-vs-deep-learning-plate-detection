# SVM Training Report

> Generated on 2026-05-23 18:09:44

## Overview

| Parameter | Value |
| --- | --- |
| Model path | `models/svm_plate.joblib` |
| Train samples | 34,093 |
| Feature dimensions | 3,780 |
| Validation samples | 8,524 |
| Class balance (train) | 17.26% positive |

## Grid Search

| Parameter | Best Value |
| --- | --- |
| C | `0.1` |
| kernel | `linear` |
| **Best CV F1** | **0.9556** |

## Validation Metrics

| Metric | Score |
| --- | --- |
| Precision | 0.9484 |
| Recall | 0.9606 |
| F1 Score | 0.9544 |

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

| | Predicted Background | Predicted Plate |
| --- | --- | --- |
| **Actual Background** | 6,975 (TN) | 77 (FP) |
| **Actual Plate** | 58 (FN) | 1,414 (TP) |
