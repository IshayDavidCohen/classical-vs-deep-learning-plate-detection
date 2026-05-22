import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from typing import Optional, Dict, Tuple, Union
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


class PlateClassification:
    """
    Wraps a sklearn pipeline + svc so that
    feature scaling is handeld transparently during both training and inference.

    we used StandardScaler because SVMs are sensitive to feature scale,
    especially with HOG features where some dimensions can dominate others.

    also class_weight balanced because there are many more background windows than plate windows
    so a balanced class weights help the SVM pay more attention to the minority class (plates)
    and not just predict everything as background.
    """
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                probability=True,
                class_weight="balanced",
            ))
        ])
        self._is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM pipeline on our feature matrix X and labels y
        Args:
            X: (n_samples, n_features) hog feature vectors
            y: (n_samples) binary labels - 1 for plate or 0 for not plate (background or sm)
        """
        self.pipeline.fit(X, y)
        self._is_fitted = True

    def grid_search(
            self,
            X: np.ndarray,
            y: np.ndarray,
            param_grid: Optional[Dict] = None,
            cv: int = 3,
            scoring: str = "f1"
    ) -> Dict:
        """
        Run cross-validation grid search and refit with the best params

        Args:
            X, y: our training data
            param_grid: a dictionary of params to search. keys have to be prefixed with '__'
            cv: number of CV folds.
            scoring: metric to optimize ('f1', 'precision', ...)

        """

        if param_grid is None:
            param_grid = {
                "svm__C": [0.1, 1, 10],
                "svm__kernel": ["linear", "rbf"],
                "svm__gamma": ["scale", "auto"],
            }

        gs = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            verbose=1,
        )
        gs.fit(X, y)
        self.pipeline = gs.best_estimator_
        self._is_fitted = True

        return {
            "best_params": gs.best_params_,
            "best_score": gs.best_score_,
        }

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and confidence scores.

        Args:
            X: (n_samples, n_features) hog feature vectors

        Returns:
            labels: (n_samples, _) prediction 0 or 1
            scores: (n_samples, _) confidence score

        This acts as the SVM decision function value, which is more reliable for ranking than probability
        """
        self._check_fitted()
        labels = self.pipeline.predict(X)
        scores = self.pipeline.decision_function(X)
        return labels, scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates (requires probability=True in SVC).

        Args:
            X: (n_samples, n_features)

        Returns:
            proba: (n_samples, 2) - columns are [P(bg), P(plate)] that it estimates
        """
        self._check_fitted()
        return self.pipeline.predict_proba(X)

    def score_windows(self, X: np.ndarray) -> np.ndarray:
        """
        Convinience method for detection.
        Returns only the decision function score for each of our window from hog
        Higher means more likely it is a plate.

        Args:
            X: (n_samples, n_features)

        Returns:
            scores: (n_windows)
        """
        self._check_fitted()
        return self.pipeline.decision_function(X)


    # Utility functions, save etc.
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted pipeline"""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PlateClassification":
        """ Load a previous saved pipeline"""
        obj = cls.__new__(cls)
        obj.pipeline = joblib.load(path)
        obj._is_fitted = True
        return obj

    # Private methods (internals) D.R.Y stuff..
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
    