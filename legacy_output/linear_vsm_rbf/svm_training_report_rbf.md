# SVM Training Report (RBF Kernel — Server Run)

> Generated on 2026-05-23 (Prometheus server)

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
| C | `10` |
| kernel | `rbf` |
| gamma | `scale` |
| **Best CV F1** | **0.9836** |

### All Results

| Kernel | C | Gamma | Fold 1 | Fold 2 | Fold 3 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| linear | 0.1 | scale | 0.954 | 0.960 | 0.953 | 0.956 |
| linear | 0.1 | auto | 0.954 | 0.960 | 0.953 | 0.956 |
| linear | 1 | scale | 0.954 | 0.960 | 0.953 | 0.956 |
| linear | 1 | auto | 0.954 | 0.960 | 0.953 | 0.956 |
| linear | 10 | scale | 0.954 | 0.960 | 0.953 | 0.956 |
| linear | 10 | auto | 0.954 | 0.960 | 0.953 | 0.956 |
| rbf | 0.1 | scale | 0.973 | 0.980 | 0.974 | 0.976 |
| rbf | 0.1 | auto | 0.973 | 0.980 | 0.974 | 0.976 |
| rbf | 1 | scale | 0.984 | 0.985 | 0.982 | 0.984 |
| rbf | 1 | auto | 0.984 | 0.985 | 0.982 | 0.984 |
| **rbf** | **10** | **scale** | **0.983** | **0.986** | **0.982** | **0.984** |
| rbf | 10 | auto | 0.983 | 0.986 | 0.982 | 0.984 |

## Validation Metrics

| Metric | Score |
| --- | --- |
| Precision | 0.99 |
| Recall | 0.98 |
| F1 Score | 0.98 |
| Accuracy | 0.99 |

## Comparison: Linear vs RBF

| Metric | Linear (C=0.1) | RBF (C=10, gamma=scale) |
| --- | --- | --- |
| Best CV F1 | 0.9556 | **0.9836** |
| Val Precision | 0.9484 | **0.99** |
| Val Recall | 0.9606 | **0.98** |
| Val F1 | 0.9544 | **0.98** |
| Training time (approx) | ~13 min/fold | ~80 min/fold |

## Notes

- RBF kernel provides a ~3 percentage point improvement in F1 over linear
- All linear configurations produced identical scores regardless of C or gamma (gamma is ignored by linear kernels)
- RBF with C=0.1 underperforms (0.976) compared to C=1 and C=10 (both ~0.984), suggesting the data benefits from less regularization with the RBF kernel
- The accuracy gain comes at significant computational cost: RBF folds took 5–8× longer than linear folds
- Run on Prometheus server (Intel i9-10800K), total wall time approximately 5 hours
