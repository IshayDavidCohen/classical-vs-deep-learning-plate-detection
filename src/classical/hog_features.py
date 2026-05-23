import cv2
import numpy as np
from skimage.feature import hog


class HOGFeatureExtractor:
    def __init__(
            self,
            target_size=(64, 128),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
    ):
        """
        Handles HOG feature extraction for license plate patches.

        Args:
            target_size (tuple): (Height, Width) to resize patches before extraction.
            orientations (int): Number of orientation bins.
            pixels_per_cell (tuple): Size (in pixels) of a cell.
            cells_per_block (tuple): Number of cells in each block.
        """
        self.target_size = target_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def compute_single(self, img_patch: np.ndarray, visualize: bool = False):
        """
        Extracts HOG feature vector from a single image patch.

        Args:
            img_patch (np.ndarray): Input BGR or Grayscale image patch.
            visualize (bool): If True, returns the HOG visualization image as well.

        Returns:
            np.ndarray: 1D feature vector (and optionally the HOG image).
        """
        # 1. Ensure grayscale
        if len(img_patch.shape) == 3:
            gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_patch.copy()

        # 2. Resize to a fixed window shape (critical for SVM)
        # cv2.resize expects (width, height)
        resized = cv2.resize(gray, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

        # 3. Extract HOG
        if visualize:
            features, hog_image = hog(
                resized,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm='L2-Hys',
                visualize=True
            )
            return features, hog_image
        else:
            features = hog(
                resized,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm='L2-Hys',
                visualize=False
            )
            return features

    def compute_batch(self, patches: list[np.ndarray]) -> np.ndarray:
        """
        Extracts HOG features from a batch of samples to build the SVM training matrix.

        Args:
            patches (list of np.ndarray): List of cropped image patches (positives and negatives).

        Returns:
            np.ndarray: 2D array of shape (num_samples, num_features) ready for SVM training.
        """
        feature_matrix = []
        for patch in patches:
            try:
                features = self.compute_single(patch, visualize=False)
                feature_matrix.append(features)
            except Exception as e:
                print(f"Skipping corrupt or empty patch: {e}")
                continue

        return np.array(feature_matrix)