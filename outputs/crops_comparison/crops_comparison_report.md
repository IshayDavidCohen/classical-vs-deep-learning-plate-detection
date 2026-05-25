# Crop-Level Comparison Report

All three models evaluated on the **exact same set of crops** from the test set.
This is an apples-to-apples comparison: same inputs, same labels, same evaluation.

> Images sampled: 200
> Total crops: 1217 (222 plates, 995 background)
> Negative crops per image: 5

## Results

| Metric | SVM Linear | SVM RBF | YOLOv8n |
| --- | --- | --- | --- |
| True Positives | 206 | 213 | 190 |
| False Negatives | 16 | 9 | 32 |
| False Positives | 9 | 2 | 57 |
| True Negatives | 986 | 993 | 938 |
| **Precision** | **0.9581** | **0.9907** | **0.7692** |
| **Recall** | **0.9279** | **0.9595** | **0.8559** |
| **F1** | **0.9428** | **0.9748** | **0.8102** |

## Interpretation

This comparison isolates the classification ability of each model.
All three receive the same pre-cropped patches, no sliding window,
no multi-scale search, no localization. The only question is:
given this patch, does it contain a license plate?

For the SVM models, this is the native task, HOG features are extracted
and the SVM classifies. For YOLO, this is an unusual setup, YOLO is
designed for full-image detection, so running it on small crops tests
whether its learned features can still recognize a plate in isolation.

## Visual Comparison

### SVM correct, YOLO wrong
These crops were classified correctly by SVM but misclassified by YOLO.
Most are likely small, tightly cropped patches where YOLO lacks the
surrounding scene context it was trained on.

![SVM wins](svm_wins.png)

### YOLO correct, SVM wrong
These crops were classified correctly by YOLO but misclassified by SVM.
Look for cases where HOG features couldn't capture the relevant pattern 
unusual plate formats, heavy blur, or textures that confuse gradient histograms.

![YOLO wins](yolo_wins.png)

## Key Takeaway

The results show a clear pattern: each model performs best in the setting it was designed for.

The SVM models were trained on exactly this kind of task - classifying isolated patches using hand-crafted gradient features. HOG captures edge structure and local texture, which is all you need when the crop is already centered on a plate. The RBF kernel pushes this even further by learning non-linear boundaries between plate and background features.

YOLO, on the other hand, was trained on full 640px images where plates appear alongside cars, roads, and scenery. It learned to detect plates in context. When we strip that context away and hand it a 40×90 pixel crop of a plate with nothing around it, the model loses the spatial cues it relies on. The same applies to background crops - a random patch of car door at crop scale doesn't look like anything YOLO learned to reason about during training.

| Setting | Best Model | F1 | Why |
| --- | --- | --- | --- |
| Crop classification | SVM RBF | 0.9748 | Built for this - HOG features on isolated patches |
| Crop classification | SVM Linear | 0.9428 | Same task, simpler decision boundary |
| Crop classification | YOLOv8n | 0.8102 | Trained on full scenes, not isolated patches |

This matters because it shows that classical methods aren't obsolete - they're specialized. When the problem is well-scoped and the features match the task, a simple pipeline can compete with or outperform a deep network. The advantage of deep learning isn't that it classifies better in every scenario, it's that it handles the full detection pipeline end-to-end without requiring manual feature engineering or sliding windows.
