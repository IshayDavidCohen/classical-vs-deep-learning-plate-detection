"""
Manim visualization: Linear vs RBF SVM decision boundaries.

Uses your actual trained models and real HOG feature data,
projected to 2D via PCA for visualization.

Install:
    pip install manim

Run:
    manim -pql svm_visualization.py SVMComparison

    -p  = preview (opens video after render)
    -ql = quality low (fast render, 480p)
    -qh = quality high (1080p, slower)

Output goes to media/videos/
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from manim import *


# ======================================================================
# Configuration — adjust paths if needed
# ======================================================================

FEATURES_PATH = "../data/features/X_train.npy"
LABELS_PATH = "../data/features/y_train.npy"
LINEAR_MODEL_PATH = "../models/svm_plate_linear.joblib"
RBF_MODEL_PATH = "../models/svm_plate_rbf.joblib"

# How many samples to plot (too many = slow render)
MAX_SAMPLES = 2000

# Grid resolution for decision boundary contour
GRID_RES = 150


# ======================================================================
# Data loading and projection
# ======================================================================

def load_and_project():
    """
    Load HOG features, subsample, PCA-project to 2D.
    Returns X_2d, y, pca, scaler for both plotting and boundary computation.
    """
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    # Subsample for speed (stratified)
    np.random.seed(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = min(len(pos_idx), MAX_SAMPLES // 5)  # keep ~1:4 ratio
    n_neg = min(len(neg_idx), MAX_SAMPLES - n_pos)

    keep = np.concatenate([
        np.random.choice(pos_idx, n_pos, replace=False),
        np.random.choice(neg_idx, n_neg, replace=False),
    ])
    X = X[keep]
    y = y[keep]

    # Scale then PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    return X_2d, y, pca, scaler


def fit_2d_svm(X_2d, y, kernel, C, gamma="scale"):
    """
    Fit a fresh SVM on the 2D PCA-projected data.
    The boundary will be geometrically correct in the projected space.
    """
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")),
    ])
    svm.fit(X_2d, y)
    return svm


def compute_decision_boundary_2d(svm_2d, X_2d, grid_res=GRID_RES):
    """
    Compute decision boundary directly in 2D space.
    """
    margin = 0.5
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_2d.decision_function(grid)
    Z = Z.reshape(xx.shape)

    return xx, yy, Z


# ======================================================================
# Manim Scenes
# ======================================================================

class SVMComparison(Scene):
    """
    Main scene: shows data points, then linear boundary, then RBF boundary.
    """

    def construct(self):
        # ---- Load data ----
        X_2d, y, pca, scaler = load_and_project()

        # Normalize to Manim coordinate space (-6 to 6 roughly)
        x_center = X_2d[:, 0].mean()
        y_center = X_2d[:, 1].mean()
        x_scale = (X_2d[:, 0].max() - X_2d[:, 0].min()) / 10
        y_scale = (X_2d[:, 1].max() - X_2d[:, 1].min()) / 6

        X_norm = np.zeros_like(X_2d)
        X_norm[:, 0] = (X_2d[:, 0] - x_center) / x_scale
        X_norm[:, 1] = (X_2d[:, 1] - y_center) / y_scale

        # ---- Title ----
        title = Text("SVM Decision Boundaries", font_size=36).to_edge(UP)
        subtitle = Text("3,780-dim HOG features → PCA → 2D projection", font_size=20, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # ---- Plot data points ----
        data_label = Text("Training Data", font_size=28).to_edge(UP)
        self.play(
            FadeOut(title), FadeOut(subtitle),
            FadeIn(data_label),
        )

        dots = VGroup()
        for i in range(len(X_norm)):
            x, y_val = X_norm[i]
            color = BLUE if y[i] == 1 else RED
            dot = Dot(
                point=np.array([x, y_val, 0]),
                radius=0.03,
                color=color,
                fill_opacity=0.6,
            )
            dots.add(dot)

        legend = VGroup(
            VGroup(Dot(color=BLUE, radius=0.06), Text("Plate", font_size=18, color=BLUE)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=RED, radius=0.06), Text("Background", font_size=18, color=RED)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(DR, buff=0.5)

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.002),
            FadeIn(legend),
            run_time=3,
        )
        self.wait(1)

        # ---- Fit 2D SVMs on projected data ----
        linear_2d = fit_2d_svm(X_2d, y, kernel="linear", C=0.1)
        rbf_2d = fit_2d_svm(X_2d, y, kernel="rbf", C=10, gamma="scale")

        # ---- Precompute both boundaries ----
        xx, yy, Z_linear = compute_decision_boundary_2d(linear_2d, X_2d)
        xx_norm = (xx - x_center) / x_scale
        yy_norm = (yy - y_center) / y_scale

        xx2, yy2, Z_rbf = compute_decision_boundary_2d(rbf_2d, X_2d)
        xx2_norm = (xx2 - x_center) / x_scale
        yy2_norm = (yy2 - y_center) / y_scale

        # ---- Show linear boundary ----
        linear_label = Text("Linear Kernel (C=0.1)", font_size=28, color=GREEN).to_edge(UP)
        f1_linear = Text("F1 = 0.9544", font_size=22, color=GREEN)
        f1_linear.next_to(linear_label, DOWN, buff=0.15)

        self.play(FadeOut(data_label), FadeIn(linear_label), FadeIn(f1_linear))

        linear_boundary = self._draw_contour(xx_norm, yy_norm, Z_linear, color=GREEN)
        self.play(Create(linear_boundary), run_time=2)
        self.wait(2)

        # ---- Transition to RBF ----
        rbf_label = Text("RBF Kernel (C=10, γ=scale)", font_size=28, color=YELLOW).to_edge(UP)
        f1_rbf = Text("F1 = 0.9836", font_size=22, color=YELLOW)
        f1_rbf.next_to(rbf_label, DOWN, buff=0.15)

        self.play(
            FadeOut(linear_label), FadeOut(f1_linear),
            FadeIn(rbf_label), FadeIn(f1_rbf),
        )

        rbf_boundary = self._draw_contour(xx2_norm, yy2_norm, Z_rbf, color=YELLOW)
        self.play(FadeOut(linear_boundary), Create(rbf_boundary), run_time=2)
        self.wait(2)

        # ---- Show both overlaid ----
        both_label = Text("Linear vs RBF", font_size=28, color=WHITE).to_edge(UP)
        self.play(FadeOut(rbf_label), FadeOut(f1_rbf), FadeIn(both_label))

        linear_boundary_2 = self._draw_contour(xx_norm, yy_norm, Z_linear, color=GREEN)
        linear_tag = Text("Linear", font_size=18, color=GREEN).to_corner(UL, buff=0.5)
        rbf_tag = Text("RBF", font_size=18, color=YELLOW).next_to(linear_tag, DOWN, buff=0.15)

        self.play(
            Create(linear_boundary_2),
            FadeIn(linear_tag), FadeIn(rbf_tag),
            run_time=1.5,
        )
        self.wait(1)

        # ---- Final insight ----
        insight = VGroup(
            Text("Same data, same features", font_size=24),
            Text("Different boundary → Different accuracy", font_size=24),
            Text("Linear F1=0.954  |  RBF F1=0.984", font_size=22, color=GRAY),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.5)

        bg_rect = BackgroundRectangle(insight, color=BLACK, fill_opacity=0.8, buff=0.2)
        self.play(FadeIn(bg_rect), Write(insight), run_time=2)
        self.wait(3)

        # ---- Fade out ----
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)

    def _draw_contour(self, xx, yy, Z, color=WHITE, level=0):
        """
        Draw the Z=0 contour as a Manim path.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        contour = ax.contour(xx, yy, Z, levels=[level])
        plt.close(fig)

        paths = VGroup()
        for seg in contour.allsegs[0]:
            if len(seg) < 2:
                continue
            points = [np.array([v[0], v[1], 0]) for v in seg]
            line = VMobject()
            line.set_points_smoothly(points)
            line.set_stroke(color=color, width=3)
            paths.add(line)

        return paths


class DataOverview(Scene):
    """
    Quick intro scene: show the numbers.
    """

    def construct(self):
        title = Text("License Plate Detection", font_size=40).to_edge(UP)
        self.play(Write(title))

        stats = VGroup(
            Text("10,125 images", font_size=28),
            Text("42,617 HOG feature vectors (3,780 dims each)", font_size=24),
            Text("17% plates / 83% background", font_size=24),
        ).arrange(DOWN, buff=0.3).center()

        self.play(FadeIn(stats, shift=UP), run_time=2)
        self.wait(2)

        # Pipeline comparison
        self.play(FadeOut(stats))

        classical = VGroup(
            Text("Classical Pipeline", font_size=28, color=GREEN),
            Text("Image → HOG → SVM → Boxes", font_size=22),
            Text("Hand-crafted features", font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.15)

        deep = VGroup(
            Text("Deep Learning Pipeline", font_size=28, color=BLUE),
            Text("Image → YOLOv8n → Boxes", font_size=22),
            Text("Learned features", font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.15)

        vs = Text("vs", font_size=36, color=YELLOW)

        comparison = VGroup(classical, vs, deep).arrange(RIGHT, buff=1).center()
        self.play(FadeIn(comparison, shift=UP), run_time=2)
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])


class ResultsReveal(Scene):
    """
    Final results scene: the three-layer comparison.
    """

    def construct(self):
        title = Text("Results", font_size=40).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Crop results
        crop_title = Text("Evaluation 1: Crop Classification", font_size=28).next_to(title, DOWN, buff=0.5)
        crop_result = VGroup(
            Text("SVM RBF:  F1 = 0.975", font_size=24, color=GREEN),
            Text("YOLOv8n:  F1 = 0.810", font_size=24, color=RED),
            Text("Winner: SVM", font_size=22, color=GREEN),
        ).arrange(DOWN, buff=0.15).next_to(crop_title, DOWN, buff=0.3)

        self.play(FadeIn(crop_title), run_time=0.5)
        self.play(FadeIn(crop_result, shift=UP), run_time=1.5)
        self.wait(2)

        # Detection results
        self.play(FadeOut(crop_title), FadeOut(crop_result))

        det_title = Text("Evaluation 2: Full-Image Detection", font_size=28).next_to(title, DOWN, buff=0.5)
        det_result = VGroup(
            Text("SVM RBF:  F1 = 0.346", font_size=24, color=RED),
            Text("YOLOv8n:  F1 = 0.947", font_size=24, color=GREEN),
            Text("Winner: YOLO (7,788x faster)", font_size=22, color=GREEN),
        ).arrange(DOWN, buff=0.15).next_to(det_title, DOWN, buff=0.3)

        self.play(FadeIn(det_title), run_time=0.5)
        self.play(FadeIn(det_result, shift=UP), run_time=1.5)
        self.wait(2)

        # Key insight
        self.play(FadeOut(det_title), FadeOut(det_result))

        insight = VGroup(
            Text("Classical ML isn't obsolete", font_size=30),
            Text("It's specialized", font_size=30, color=YELLOW),
            Text("", font_size=10),
            Text("The SVM classifies crops better than YOLO.", font_size=22),
            Text("But detection requires localization —", font_size=22),
            Text("and that's where end-to-end learning wins.", font_size=22),
        ).arrange(DOWN, buff=0.15).center()

        self.play(Write(insight), run_time=3)
        self.wait(4)

        self.play(*[FadeOut(mob) for mob in self.mobjects])