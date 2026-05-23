"""
Train the SVM classifier on HOG features.

This script demonstrates the full training workflow:
1. Load HOG features (from .npy files your teammate produces)
2. Split into train / val
3. Train SVM (with optional grid search)
4. Evaluate on validation set
5. Save the trained model

Run:
    python -m src.classical.train_svm

Expects your teammate to have saved features like:
    data/features/X_train.npy   — shape (n_samples, n_features)
    data/features/y_train.npy   — shape (n_samples,)
    data/features/X_val.npy     — (optional, for evaluation)
    data/features/y_val.npy     — (optional)

If val files don't exist, a random split is made from the training data.
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.classical.svm_classifier import PlateClassification
from src.evaluation.metrics import classification_metrics


def load_features(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load feature arrays from disk."""
    data_dir = Path(data_dir)

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    X_val, y_val = None, None
    if (data_dir / "X_val.npy").exists():
        X_val = np.load(data_dir / "X_val.npy")
        y_val = np.load(data_dir / "y_val.npy")

    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser(description="Train SVM on HOG features")
    parser.add_argument("--data-dir", type=str, default="data/features",
                        help="Directory containing X_train.npy, y_train.npy, etc.")
    parser.add_argument("--model-path", type=str, default="models/svm_plate.joblib",
                        help="Where to save the trained model.")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run grid search for hyperparameter tuning.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Val split ratio if no separate val set exists.")
    args = parser.parse_args()

    # ---- Load data ----
    print(f"Loading features from {args.data_dir} ...")
    X_train, y_train, X_val, y_val = load_features(args.data_dir)

    # Split if no val set
    if X_val is None:
        print(f"No validation set found — splitting {args.test_size:.0%} from training data.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=args.test_size, random_state=42, stratify=y_train,
        )

    print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Class balance (train): {np.mean(y_train):.2%} positive")

    # ---- Train ----
    clf = PlateClassification()

    if args.grid_search:
        print("\nRunning grid search ...")
        result = clf.grid_search(X_train, y_train)
        print(f"Best params: {result['best_params']}")
        print(f"Best CV F1:  {result['best_score']:.4f}")
    else:
        print("\nTraining SVM ...")
        clf.train(X_train, y_train)

    # ---- Evaluate ----
    print("\n--- Validation results ---")
    preds, scores = clf.predict(X_val)
    metrics = classification_metrics(y_val, preds)

    # ---- Save ----
    clf.save(args.model_path)
    print(f"\nModel saved to {args.model_path}")


if __name__ == "__main__":
    main()