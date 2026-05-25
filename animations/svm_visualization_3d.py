"""
3D Manim visualization: Linear vs RBF SVM decision boundaries.

Run:
    manim -pql svm_visualization_3d.py SVMComparison3D
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from manim import *


FEATURES_PATH = "../data/features/X_train.npy"
LABELS_PATH = "../data/features/y_train.npy"

MAX_SAMPLES = 500
GRID_RES = 30


def load_and_project_3d():
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    np.random.seed(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = min(len(pos_idx), MAX_SAMPLES // 5)
    n_neg = min(len(neg_idx), MAX_SAMPLES - n_pos)

    keep = np.concatenate([
        np.random.choice(pos_idx, n_pos, replace=False),
        np.random.choice(neg_idx, n_neg, replace=False),
    ])
    X = X[keep]
    y = y[keep]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X_scaled)

    return X_3d, y


def fit_2d_svm_on_3d(X_3d, y, kernel, C, gamma="scale"):
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced")),
    ])
    svm.fit(X_3d, y)
    return svm


def compute_boundary_surface(svm, X_3d, grid_res=GRID_RES):
    margin = 0.5
    x_min, x_max = X_3d[:, 0].min() - margin, X_3d[:, 0].max() + margin
    y_min, y_max = X_3d[:, 1].min() - margin, X_3d[:, 1].max() + margin
    z_min, z_max = X_3d[:, 2].min() - margin, X_3d[:, 2].max() + margin

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    z_vals = np.linspace(z_min, z_max, grid_res)

    def find_crossing(x_val, y_val):
        test_points = np.column_stack([
            np.full(len(z_vals), x_val),
            np.full(len(z_vals), y_val),
            z_vals,
        ])
        decisions = svm.decision_function(test_points)
        for k in range(len(decisions) - 1):
            if decisions[k] * decisions[k + 1] < 0:
                t = decisions[k] / (decisions[k] - decisions[k + 1])
                return [x_val, y_val, z_vals[k] + t * (z_vals[k + 1] - z_vals[k])]
        return None

    print(f"  Computing RBF surface ({grid_res}x{grid_res} grid, parallelized) ...")
    results = Parallel(n_jobs=-1)(
        delayed(find_crossing)(xx[i, j], yy[i, j])
        for i in range(grid_res) for j in range(grid_res)
    )

    surface_points = [r for r in results if r is not None]
    print(f"  Found {len(surface_points)} surface points")
    return np.array(surface_points) if surface_points else None


class SVMComparison3D(ThreeDScene):
    def construct(self):
        X_3d, y = load_and_project_3d()

        center = X_3d.mean(axis=0)
        scale = (X_3d.max(axis=0) - X_3d.min(axis=0)) / 8
        X_norm = (X_3d - center) / scale

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # ---- Title ----
        title = Text("SVM Decision Boundaries in 3D", font_size=32)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)

        subtitle = Text("3,780-dim HOG features → PCA → 3D", font_size=18, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(subtitle)

        self.play(Write(title), FadeIn(subtitle))
        self.wait(1)

        # ---- Axes ----
        axes = ThreeDAxes(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1], z_range=[-4, 4, 1],
            x_length=10, y_length=10, z_length=8,
        )
        axes.set_stroke(opacity=0.3)
        self.play(Create(axes), run_time=1)

        # ---- Data points ----
        data_label = Text("Training Data", font_size=24)
        data_label.to_edge(UP)
        self.add_fixed_in_frame_mobjects(data_label)

        self.play(FadeOut(title), FadeOut(subtitle), FadeIn(data_label))

        dots = VGroup()
        for i in range(len(X_norm)):
            x, y_val, z = X_norm[i]
            color = BLUE if y[i] == 1 else RED
            dot = Dot3D(point=np.array([x, y_val, z]), radius=0.03, color=color, fill_opacity=0.5)
            dots.add(dot)

        legend = VGroup(
            VGroup(Dot(color=BLUE, radius=0.06), Text("Plate", font_size=16, color=BLUE)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(color=RED, radius=0.06), Text("Background", font_size=16, color=RED)).arrange(RIGHT, buff=0.1),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.to_corner(DR, buff=0.4)
        self.add_fixed_in_frame_mobjects(legend)

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.002),
            FadeIn(legend),
            run_time=3,
        )

        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(1.5)

        # ---- Fit SVMs (happens during rotation wait) ----
        linear_svm = fit_2d_svm_on_3d(X_3d, y, kernel="linear", C=0.1)
        rbf_svm = fit_2d_svm_on_3d(X_3d, y, kernel="rbf", C=10, gamma="scale")

        # ---- Linear boundary (plane) ----
        linear_label = Text("Linear Kernel (C=0.1)", font_size=24, color=GREEN)
        linear_label.to_edge(UP)
        self.add_fixed_in_frame_mobjects(linear_label)
        f1_linear = Text("F1 = 0.9544", font_size=18, color=GREEN)
        f1_linear.next_to(linear_label, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(f1_linear)

        self.play(FadeOut(data_label), FadeIn(linear_label), FadeIn(f1_linear))

        svm_model = linear_svm.named_steps["svm"]
        svm_scaler = linear_svm.named_steps["scaler"]
        w = svm_model.coef_[0]
        b = svm_model.intercept_[0]

        def linear_plane_func(u, v):
            x_orig = u * scale[0] + center[0]
            y_orig = v * scale[1] + center[1]
            x_sc = (x_orig - svm_scaler.mean_[0]) / svm_scaler.scale_[0]
            y_sc = (y_orig - svm_scaler.mean_[1]) / svm_scaler.scale_[1]
            z_sc = -(w[0] * x_sc + w[1] * y_sc + b) / w[2]
            z_orig = z_sc * svm_scaler.scale_[2] + svm_scaler.mean_[2]
            z_norm = (z_orig - center[2]) / scale[2]
            z_norm = np.clip(z_norm, -4, 4)
            return np.array([u, v, z_norm])

        linear_surface = Surface(
            linear_plane_func, u_range=[-4.5, 4.5], v_range=[-4.5, 4.5],
            resolution=(20, 20), fill_opacity=0.25, fill_color=GREEN,
            stroke_color=GREEN, stroke_width=1, stroke_opacity=0.5,
        )

        self.play(Create(linear_surface), run_time=2)
        self.wait(1.5)

        # ---- RBF boundary (surface) ----
        rbf_label = Text("RBF Kernel (C=10, γ=scale)", font_size=24, color=YELLOW)
        rbf_label.to_edge(UP)
        self.add_fixed_in_frame_mobjects(rbf_label)
        f1_rbf = Text("F1 = 0.9836", font_size=18, color=YELLOW)
        f1_rbf.next_to(rbf_label, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(f1_rbf)

        self.play(
            FadeOut(linear_label), FadeOut(f1_linear),
            FadeIn(rbf_label), FadeIn(f1_rbf),
        )

        surface_pts = compute_boundary_surface(rbf_svm, X_3d)

        rbf_dots = VGroup()
        if surface_pts is not None and len(surface_pts) > 0:
            surf_norm = (surface_pts - center) / scale
            for pt in surf_norm:
                dot = Dot3D(point=pt, radius=0.02, color=YELLOW, fill_opacity=0.3)
                rbf_dots.add(dot)

        self.play(
            FadeOut(linear_surface),
            LaggedStart(*[FadeIn(d) for d in rbf_dots], lag_ratio=0.002),
            run_time=2,
        )
        self.wait(1.5)

        # ---- Show both ----
        both_label = Text("Linear vs RBF", font_size=24, color=WHITE)
        both_label.to_edge(UP)
        self.add_fixed_in_frame_mobjects(both_label)

        linear_tag = Text("Linear (plane)", font_size=16, color=GREEN)
        linear_tag.to_corner(UL, buff=0.4)
        self.add_fixed_in_frame_mobjects(linear_tag)
        rbf_tag = Text("RBF (surface)", font_size=16, color=YELLOW)
        rbf_tag.next_to(linear_tag, DOWN, buff=0.1)
        self.add_fixed_in_frame_mobjects(rbf_tag)

        self.play(
            FadeOut(rbf_label), FadeOut(f1_rbf),
            FadeIn(both_label), FadeIn(linear_tag), FadeIn(rbf_tag),
        )

        linear_surface_2 = Surface(
            linear_plane_func, u_range=[-4.5, 4.5], v_range=[-4.5, 4.5],
            resolution=(20, 20), fill_opacity=0.2, fill_color=GREEN,
            stroke_color=GREEN, stroke_width=1, stroke_opacity=0.4,
        )
        self.play(Create(linear_surface_2), run_time=1.5)
        self.wait(1)

        # ---- Final insight ----
        insight = VGroup(
            Text("Linear: flat plane - cannot follow curved boundaries", font_size=20),
            Text("RBF: flexible surface - wraps around the plate cluster", font_size=20),
            Text("Linear F1=0.954  |  RBF F1=0.984", font_size=18, color=GRAY),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.4)

        bg_rect = BackgroundRectangle(insight, color=BLACK, fill_opacity=0.85, buff=0.15)
        self.add_fixed_in_frame_mobjects(bg_rect)
        self.add_fixed_in_frame_mobjects(insight)

        self.play(FadeIn(bg_rect), Write(insight), run_time=2)
        self.wait(2)

        self.stop_ambient_camera_rotation()
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)