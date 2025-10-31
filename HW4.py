"""
CAP5610 HW4 (DROP version, no CLI): KNN, SVM (Linear/Poly/RBF), KMeans on lncRNA_5_Cancers.csv
- 5-fold Stratified CV
- Macro Accuracy/Precision/Recall/F1, ROC-AUC, PR-AUC
- Confusion matrices and OvR ROC/PR figures
- KMeans for K=2..7 with PCA visualization, Elbow, Silhouette
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,confusion_matrix,roc_curve,auc,average_precision_score,roc_auc_score,precision_recall_curve,silhouette_score,)

from umap import UMAP

# -----------------------------
# Config (edit here if needed)
# -----------------------------
RANDOM_STATE = 42
N_SPLITS = 5
CSV_PATH = "lncRNA_5_Cancers.csv"        # put your CSV here
OUT_DIR = "results"                      # output folder
POLY_DEGREE = 2                          # default polynomial degree

# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs(out_dir: str) -> Tuple[str, str]:
    """Create output and figures directories if needed."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return out_dir, fig_dir

def load_data(auto_path: str = CSV_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and infer label column (label/cancer/cancer_type/class or last column)."""
    if not os.path.exists(auto_path):
        raise FileNotFoundError(
            f"Dataset not found: {auto_path}\n"
            f"Please place 'lncRNA_5_Cancers.csv' inside a 'data' folder next to this script, "
            f"or update CSV_PATH at the top."
        )
    df = pd.read_csv(auto_path)
    label_col_candidates = [c for c in df.columns if c.lower() in ("label", "cancer", "cancer_type", "class")]
    label_col = label_col_candidates[0] if label_col_candidates else df.columns[-1]
    y = df[label_col].astype(str)
    X = df.drop(columns=[label_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    print(f" Loaded dataset '{auto_path}' with shape {X.shape} and label column '{label_col}'.")
    return X, y
@dataclass
class CVResults:
    name: str
    y_true_all: List[str]
    y_pred_all: List[str]
    scores_all: np.ndarray  # shape (n_samples, n_classes)
    acc: List[float]
    prec_macro: List[float]
    rec_macro: List[float]
    f1_macro: List[float]
    roc_auc_macro: List[float]
    pr_auc_macro: List[float]

def compute_multiclass_auc(y_true: np.ndarray, y_score: np.ndarray, classes: List[str]) -> Tuple[float, float]:
    """
    Compute macro ROC-AUC and macro PR-AUC using one-vs-rest binarization.
    y_true: shape (n_samples,)
    y_score: shape (n_samples, n_classes) decision_function or predict_proba scores
    """
    y_bin = label_binarize(y_true, classes=classes)
    roc_auc = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
    pr_auc = average_precision_score(y_bin, y_score, average="macro")
    return float(roc_auc), float(pr_auc)


def trimmed_inlier_mask(X: np.ndarray, fraction_to_trim: float = 0.05) -> np.ndarray:
    """Return a boolean mask marking inliers after trimming the most distant samples.

    The mask is built using squared Euclidean distance to the feature-wise median,
    which makes it robust to extreme values compared to the mean.  The farthest
    ``fraction_to_trim`` samples are flagged as outliers.  When ``fraction_to_trim``
    is zero or the dataset is too small, the function falls back to keeping all
    samples.
    """

    if X.ndim != 2:
        raise ValueError("trimmed_inlier_mask expects a 2D array of shape (n_samples, n_features)")

    n_samples = X.shape[0]
    if n_samples == 0:
        return np.zeros(0, dtype=bool)

    fraction_to_trim = float(np.clip(fraction_to_trim, 0.0, 0.49))
    # If trimming would discard fewer than one sample, keep everything.
    if fraction_to_trim == 0.0 or n_samples * fraction_to_trim < 1.0:
        return np.ones(n_samples, dtype=bool)

    median = np.median(X, axis=0)
    squared_dist = np.sum((X - median) ** 2, axis=1)
    threshold = np.quantile(squared_dist, 1.0 - fraction_to_trim)
    mask = squared_dist <= threshold

    # Guard against degenerate cases (e.g., all points identical)
    if mask.sum() == 0:
        mask[:] = True

    return mask

def run_cv_pipeline(name: str, estimator, X: pd.DataFrame, y: pd.Series, classes: List[str]) -> CVResults:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all = [], []
    scores_list = []
    accs, precs, recs, f1s, roc_aucs, pr_aucs = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Always scale before SVM/KNN
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", estimator)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        # Macro metrics
        accs.append(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        precs.append(prec); recs.append(rec); f1s.append(f1)

        # Scores: prefer decision_function (fast; no probability calibration)
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "decision_function"):
            scores = pipe.decision_function(X_test)
            if scores.ndim == 1:  # binary shape fix
                scores = np.vstack([-scores, scores]).T
        elif hasattr(clf, "predict_proba"):
            scores = pipe.predict_proba(X_test)  # KNN path
        else:
            # Fallback (not ideal for ROC/PR): 1.0 for predicted class
            scores = np.zeros((len(y_test), len(classes)), dtype=float)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            for i, pred in enumerate(y_pred):
                scores[i, class_to_idx[pred]] = 1.0

        roc_auc, pr_auc = compute_multiclass_auc(y_test.values, scores, classes=classes)
        roc_aucs.append(roc_auc); pr_aucs.append(pr_auc)
        scores_list.append(scores)

        print(f"[{name}] Fold {fold}: acc={accs[-1]:.3f}, prec={prec:.3f}, rec={rec:.3f}, "
              f"f1={f1:.3f}, ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")

    scores_all = np.vstack(scores_list) if scores_list else np.empty((0, len(classes)))
    return CVResults(
        name=name,
        y_true_all=y_true_all,
        y_pred_all=y_pred_all,
        scores_all=scores_all,
        acc=accs,
        prec_macro=precs,
        rec_macro=recs,
        f1_macro=f1s,
        roc_auc_macro=roc_aucs,
        pr_auc_macro=pr_aucs
    )

def summarize_cv(res: CVResults) -> pd.Series:
    return pd.Series({
        "Accuracy (mean)": np.mean(res.acc),
        "Precision_macro (mean)": np.mean(res.prec_macro),
        "Recall_macro (mean)": np.mean(res.rec_macro),
        "F1_macro (mean)": np.mean(res.f1_macro),
        "ROC-AUC_macro (mean)": np.mean(res.roc_auc_macro),
        "PR-AUC_macro (mean)": np.mean(res.pr_auc_macro),
    }, name=res.name)

# NEW: helper to save one summary row per model
def save_all_summaries(knn_res: CVResults, svm_results: Dict[str, CVResults], out_dir: str):
    """
    Build a single CSV with one row per model (KNN + each SVM).
    """
    all_results = {"KNN": knn_res}
    all_results.update(svm_results)

    rows = []
    for name, res in all_results.items():
        s = summarize_cv(res)
        s.name = name  # ensure row name is the model name
        rows.append(s.to_frame().T)

    summary_df = pd.concat(rows, axis=0)
    summary_df.index.name = "Model"
    summary_df.to_csv(os.path.join(out_dir, "hw4_knn_svm_summary.csv"))
    print(" Saved model summary:", os.path.join(out_dir, "hw4_knn_svm_summary.csv"))

def plot_class_counts_from_y(y: pd.Series, out_dir: str, fig_dir: str, title: str = "Class Counts", file_stem: str = "class_counts"):
    """
    ONE bar per class. Counts how many rows belong to each class in `y`.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    y_str = y.astype(str)

    # Keep stable order of first appearance in the data
    class_order = y_str.drop_duplicates().tolist()
    counts = y_str.value_counts().reindex(class_order).fillna(0).astype(int)

    # Save counts table
    counts_path = os.path.join(out_dir, f"{file_stem}.csv")
    counts.to_frame("count").to_csv(counts_path, index=True)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(counts)), counts.values)

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")

    # Add count labels above each bar
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    fig.tight_layout()
    png_path = os.path.join(fig_dir, f"{file_stem}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f" Saved counts table: {counts_path}")
    print(f" Saved bar chart:   {png_path}")
    return counts

def plot_confusion(y_true: List[str], y_pred: List[str], classes: List[str], title: str, outpath: str):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_multiclass_roc_pr(y_true: List[str], scores: np.ndarray, classes: List[str], base_name: str, fig_dir: str):
    """
    Plot OvR ROC and PR curves (per-class only).
    Saves two figures: *_roc.png and *_pr.png
    """
    y_bin = label_binarize(np.array(y_true), classes=classes)
    n_classes = len(classes)

    # ---------- ROC (per class only) ----------
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC={roc_auc[i]:.2f})")
    ax.plot([0, 1], [0, 1], linestyle=":")
    ax.set_title(f"{base_name} – ROC (OvR)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{base_name}_roc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- PR (per class only) ----------
    pr, rc, pr_ap = {}, {}, {}
    for i in range(n_classes):
        pr[i], rc[i], _ = precision_recall_curve(y_bin[:, i], scores[:, i])
        pr_ap[i] = average_precision_score(y_bin[:, i], scores[:, i])

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n_classes):
        ax.plot(rc[i], pr[i], label=f"{classes[i]} (AP={pr_ap[i]:.2f})")
    ax.set_title(f"{base_name} – PR (OvR)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{base_name}_pr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> np.ndarray:
    """
    One-vs-rest accuracy for each class:
      accuracy_i = (TP_i + TN_i) / (TP_i + FP_i + TN_i + FN_i)
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    per_acc = []
    N = cm.sum()
    for i in range(len(classes)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = N - (TP + FP + FN)
        per_acc.append((TP + TN) / N if N else 0.0)
    return np.array(per_acc, dtype=float)

def save_per_class_table(model_name: str,
                         y_true_all: List[str],
                         y_pred_all: List[str],
                         classes: List[str],
                         out_dir: str):
    """
    Writes {out_dir}/per_class_metrics/{model_name}_per_class.csv
    with columns: Accuracy, Precision, Recall, F1, Support (per class).
    """
    y_true_arr = np.asarray(y_true_all)
    y_pred_arr = np.asarray(y_pred_all)

    # per-class Precision/Recall/F1/Support
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0
    )

    # per-class one-vs-rest Accuracy
    acc = _per_class_accuracy(y_true_arr, y_pred_arr, classes)

    df = pd.DataFrame({
        "Class": classes,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Support": support.astype(int)
    })
    df = df.set_index("Class")

    out_sub = os.path.join(out_dir, "per_class_metrics")
    os.makedirs(out_sub, exist_ok=True)
    out_path = os.path.join(out_sub, f"{model_name}_per_class.csv")
    df.to_csv(out_path)
    print(f"✅ Per-class metrics saved: {out_path}")

    # ===============================================================
    # HELPER FUNCTION: Visualization for Task 3
    # ===============================================================

def plot_task3_visualizations( X_vis,y_true, classes,cids, k, fig_dir, prefix, method_name, figsize=(9, 6), legend_out=True, centroid_override_2d=None, ):
    """
    Single figure ONLY:
      • Points = data, colored by TRUE labels
      • Shapes = cluster assignment (different marker per cluster)
      • Centroids (X) = optionally overridden by centroid_override_2d
    """
    os.makedirs(fig_dir, exist_ok=True)

    X_vis = np.asarray(X_vis)
    if X_vis.ndim != 2 or X_vis.shape[1] != 2:
        raise ValueError(f"{prefix}/{method_name}: X_vis must be (n,2); got {X_vis.shape}")

    # Stable color map for true labels
    cmap = plt.get_cmap("tab10", max(len(classes), 5))
    color_map = {cls: cmap(i % cmap.N) for i, cls in enumerate(classes)}
    y_arr = np.asarray(y_true)

    # Markers by cluster
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    # If no override centroids, compute standard mean in 2D
    if centroid_override_2d is None:
        centroids_2d = []
        for cid in range(k):
            m = (cids == cid)
            if np.any(m):
                centroids_2d.append(X_vis[m].mean(axis=0))
        centroids_2d = np.array(centroids_2d)
    else:
        centroids_2d = np.asarray(centroid_override_2d)
        if centroids_2d.shape != (k, 2):
            raise ValueError(f"{prefix}/{method_name}: centroid_override_2d must be (k,2); got {centroids_2d.shape}")

    # -------- CLUSTERS plot (colors = true labels; shapes = clusters) --------
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    for cid in range(k):
        mask = (cids == cid)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        colors = [color_map[y_arr[i]] for i in idxs]
        ax.scatter(
            X_vis[mask, 0], X_vis[mask, 1],
            s=28, alpha=0.85, c=colors,
            marker=markers[cid % len(markers)],
            edgecolor='black', linewidths=0.4,
            label=f"Cluster {cid+1}"
        )

    if centroids_2d.size > 0:
        ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            s=170, c='red', marker='X',
            edgecolor='white', linewidths=0.7,
            label='Centroid'
        )

    ax.set_title(f"{prefix} (K={k}) – {method_name} shapes = clusters, colors = true labels")
    ax.set_xlabel(f"{method_name}-1"); ax.set_ylabel(f"{method_name}-2")

    # Legends: keep TRUE-LABEL color legend + cluster/centroid legend
    class_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cname],
               markeredgecolor='black', markersize=8, label=cname)
        for cname in classes
    ]
    cluster_handles = [Line2D([0], [0], marker=m, color='black',
                              markerfacecolor='white', markersize=8, label=f"Cluster {i+1}")
                       for i, m in enumerate(markers[:k])]
    centroid_handle = Line2D([0], [0], marker='X', color='black', markerfacecolor='red',
                             markersize=9, label='Centroid')

    if legend_out:
        # push plot left and put legends outside on the right
        plt.subplots_adjust(right=0.78)
        leg1 = ax.legend(handles=class_handles, title="True Labels",
                         fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1.0))
        ax.add_artist(leg1)
        ax.legend(handles=cluster_handles + [centroid_handle], title="Clusters",
                  fontsize=9, loc='lower left', bbox_to_anchor=(1.02, 0.0))
    else:
        leg1 = ax.legend(handles=class_handles, title="True Labels",
                         fontsize=9, loc='upper right')
        ax.add_artist(leg1)
        ax.legend(handles=cluster_handles + [centroid_handle], title="Clusters",
                  fontsize=9, loc='lower right')

    fig.tight_layout()
    out_path = os.path.join(fig_dir, f"k{k}_{method_name.lower()}_clusters.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Tasks
# -----------------------------
def task1_knn(X: pd.DataFrame, y: pd.Series, classes: List[str], out_dir: str, fig_dir: str) -> CVResults:
    print("\n=== Task 1: KNN (5-fold Stratified CV) ===")
    model = KNeighborsClassifier(n_neighbors=5)
    res = run_cv_pipeline("KNN", model, X, y, classes)
    summarize_cv(res).to_frame().T.to_csv(os.path.join(out_dir, "task1_knn_summary.csv"), index=True)
    plot_confusion(res.y_true_all, res.y_pred_all, classes,
                   "KNN – Confusion Matrix (5-fold aggregated)",
                   os.path.join(fig_dir, "task1_knn_confusion.png"))
    plot_multiclass_roc_pr(res.y_true_all, res.scores_all, classes, "task1_knn", fig_dir)
    return res

def task2_svm_drop(X: pd.DataFrame, y: pd.Series, classes: List[str], out_dir: str, fig_dir: str,
                   poly_degree: int = POLY_DEGREE) -> Dict[str, CVResults]:
    """
    Faster SVMs:
      - Linear: LinearSVC (true linear solver)
      - Poly: SVC(poly) with degree=poly_degree, probability=False
      - RBF:  SVC(rbf) with probability=False
    """
    print("\n=== Task 2: SVM Kernels (DROP/faster) – 5-fold Stratified CV ===")
    kernels = {
        "SVM_Linear": LinearSVC(random_state=RANDOM_STATE, dual="auto", max_iter=5000),
        "SVM_Poly":   SVC(kernel="poly", degree=poly_degree, gamma="scale", coef0=1.0,
                          C=1.0, probability=False, shrinking=True, cache_size=2000,
                          random_state=RANDOM_STATE),
        "SVM_RBF":    SVC(kernel="rbf", gamma="scale", C=1.0,
                          probability=False, shrinking=True, cache_size=2000,
                          random_state=RANDOM_STATE),
    }

    all_summaries = []
    results: Dict[str, CVResults] = {}

    for name, model in kernels.items():
        print(f"\n-- {name} --")
        res = run_cv_pipeline(name, model, X, y, classes)
        results[name] = res
        all_summaries.append(summarize_cv(res))

        plot_confusion(res.y_true_all, res.y_pred_all, classes,
                       f"{name} – Confusion Matrix (5-fold aggregated)",
                       os.path.join(fig_dir, f"task2_{name}_confusion.png"))
        plot_multiclass_roc_pr(res.y_true_all, res.scores_all, classes, f"task2_{name}", fig_dir)

    pd.DataFrame(all_summaries).to_csv(os.path.join(out_dir, "task2_svm_kernel_comparison.csv"), index=True)
    return results

def task3_kmeans(X: pd.DataFrame, y: pd.Series, classes: list, out_dir: str, fig_dir: str):
    """
    Task 3: KMeans Clustering (K=2..7)
      • V1: KMeans on standardized original features (no PCA for clustering)
            -> Visualize in PCA(2); centroids = robust medoids in 2D (resistant to outliers).
      • V2: KMeans on top-5,000 variance features + PCA(100) for clustering
            -> Visualize in PCA(2) and UMAP(2); centroids = robust medoids in 2D.

    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---------- local helpers ----------
    def plot_metric(k_values, values, title, ylabel, path, color=None, marker="o"):
        fig, ax = plt.subplots(figsize=(6, 4))
        if color is None:
            ax.plot(k_values, values, marker=marker)
        else:
            ax.plot(k_values, values, marker=marker, color=color)
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    def medoid_centroids_2d(X2d: np.ndarray, cids: np.ndarray, k: int) -> np.ndarray:
        """
        Robust centroids in 2D via cluster medoids (actual points minimizing total intra-cluster distance).
        Immune to outliers pulling the centroid.
        """
        if valid_mask is None:
            valid_mask = np.ones(len(X2d), dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape[0] != X2d.shape[0]:
                raise ValueError("valid_mask must have the same length as X2d")

        C = np.full((k, 2), np.nan, dtype=float)
        for cid in range(k):
            cluster_mask = (cids == cid)
            inlier_points = X2d[cluster_mask & valid_mask]
            if inlier_points.shape[0] == 0:
                # Fall back to all points from the cluster (may include outliers)
                P = X2d[cluster_mask]
            else:
                P = inlier_points

            if P.shape[0] == 0:
                continue
            if P.shape[0] == 1:
                C[cid] = P[0]
                continue
            D = cdist(P, P, metric="euclidean")
            idx = np.argmin(D.sum(axis=1))
            C[cid] = P[idx]
        # fill any nan clusters with plain means (rare: empty cluster)
        nan_rows = np.isnan(C).any(axis=1)
        if nan_rows.any():
            for cid in np.where(nan_rows)[0]:
                P = X2d[cids == cid]
                if P.size:
                    C[cid] = P.mean(axis=0)
                else:
                    C[cid] = np.array([np.nan, np.nan], dtype=float)
        return C

    # Because some users haven't updated the helper signature yet, we adapt at runtime:
    def _plot_with_centroids_safe(X_vis, y, classes, cids, k, this_fig_dir, prefix, method_name, C2d):
        try:
            # New signature (with centroid_override_2d, legend_out, figsize)
            plot_task3_visualizations(
                X_vis, y, classes, cids, k,
                this_fig_dir, prefix, method_name,
                figsize=(9, 6), legend_out=True,
                centroid_override_2d=C2d
            )
        except TypeError:
            # Old signature (no centroid_override_2d); fall back to default behavior
            plot_task3_visualizations(
                X_vis, y, classes, cids, k,
                this_fig_dir, prefix, method_name
            )

    # ---------- common config ----------
    k_values = list(range(2, 8))
    RANDOM_STATE = 42

    # ===============================================================
    # VERSION 1 — KMeans on standardized original features
    # ===============================================================
    print("\n=== Task 3 (V1): KMeans on standardized original features ===")
    v1_out, v1_fig = os.path.join(out_dir, "v1"), os.path.join(fig_dir, "v1")
    os.makedirs(v1_out, exist_ok=True)
    os.makedirs(v1_fig, exist_ok=True)

    scaler = StandardScaler()
    Xs_v1 = scaler.fit_transform(X)

    inlier_mask_v1 = trimmed_inlier_mask(Xs_v1, fraction_to_trim=0.05)
    if inlier_mask_v1.sum() <= max(k_values):
        # Not enough samples remain to support the full range of K; keep everything.
        inlier_mask_v1 = np.ones_like(inlier_mask_v1, dtype=bool)
        print(" [V1] Skipping trimming because it would remove too many samples.")
    else:
        removed = (~inlier_mask_v1).sum()
        print(f" [V1] Trimmed {removed} potential outliers before clustering (fraction=5%).")

    # PCA(2) for visualization only
    pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d_v1 = pca_vis.fit_transform(Xs_v1)

    inertias_v1, silhouettes_v1 = [], []

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=50,                 # more restarts for stability
            random_state=RANDOM_STATE
        )
        X_fit = Xs_v1[inlier_mask_v1]
        cids_inliers = km.fit_predict(X_fit)
        cids_all = km.predict(Xs_v1)

        # Robust centroids in the SAME 2D plotting space (medoids)
        centroids_2d = medoid_centroids_2d(X2d_v1, cids_all, k, valid_mask=inlier_mask_v1)

        # Single clusters+labels plot (your helper should now only generate this one)
        _plot_with_centroids_safe(
            X2d_v1, y, classes, cids_all, k,
            v1_fig, "V1 KMeans", "PCA2", centroids_2d
        )

        inertias_v1.append(km.inertia_)
        try:
            silhouettes_v1.append(silhouette_score(X_fit, cids_inliers))
        except ValueError:
            silhouettes_v1.append(np.nan)

    plot_metric(k_values, inertias_v1,   "Elbow Method (V1)",      "Inertia",
                os.path.join(v1_fig, "elbow.png"),      color="tab:blue",  marker="o")
    plot_metric(k_values, silhouettes_v1,"Silhouette Score (V1)", "Silhouette",
                os.path.join(v1_fig, "silhouette.png"), color="tab:orange", marker="s")
    pd.DataFrame({"K": k_values, "Inertia": inertias_v1, "Silhouette": silhouettes_v1}).to_csv(
        os.path.join(v1_out, "metrics.csv"), index=False)
    print("✅ V1 results saved successfully.")

    # ===============================================================
    # VERSION 2 — Top 5000 variance + PCA(100) + UMAP(2)
    # ===============================================================
    print("\n=== Task 3 (V2): KMeans with Feature Selection + PCA(100) + UMAP ===")
    v2_out, v2_fig = os.path.join(out_dir, "v2"), os.path.join(fig_dir, "v2")
    os.makedirs(v2_out, exist_ok=True)
    os.makedirs(v2_fig, exist_ok=True)

    print("Selecting top 5000 high-variance features…")
    variances = X.var(axis=0)
    top_genes = variances.nlargest(5000).index
    X_sel = X[top_genes]

    scaler = StandardScaler()
    Xs_v2 = scaler.fit_transform(X_sel)

    # PCA(100) for clustering
    pca_100 = PCA(n_components=100, random_state=RANDOM_STATE)
    X_pca100 = pca_100.fit_transform(Xs_v2)

    inlier_mask_v2 = trimmed_inlier_mask(X_pca100, fraction_to_trim=0.05)
    if inlier_mask_v2.sum() <= max(k_values):
        inlier_mask_v2 = np.ones_like(inlier_mask_v2, dtype=bool)
        print(" [V2] Skipping trimming because it would remove too many samples.")
    else:
        removed_v2 = (~inlier_mask_v2).sum()
        print(f" [V2] Trimmed {removed_v2} potential outliers before clustering (fraction=5%).")

    # Visualization embeddings (PCA2 + UMAP2)
    print("Computing visualization embeddings…")
    pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d_pca = pca_2.fit_transform(X_pca100)
    umap_2 = UMAP(n_components=2, random_state=RANDOM_STATE)
    X2d_umap = umap_2.fit_transform(X_pca100)

    inertias_v2, silhouettes_v2 = [], []

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=50,
            random_state=RANDOM_STATE
        )
        X_fit_v2 = X_pca100[inlier_mask_v2]
        cids_inliers_v2 = km.fit_predict(X_fit_v2)
        cids_all_v2 = km.predict(X_pca100)

        # Robust centroids in each 2D space (medoids)
        centroids_pca2 = medoid_centroids_2d(X2d_pca, cids_all_v2, k, valid_mask=inlier_mask_v2)
        centroids_umap2 = medoid_centroids_2d(X2d_umap, cids_all_v2, k, valid_mask=inlier_mask_v2)

        # PCA2 view
        _plot_with_centroids_safe(
            X2d_pca, y, classes, cids_all_v2, k,
            v2_fig, "V2 KMeans", "PCA2", centroids_pca2
        )

        # UMAP2 view
        _plot_with_centroids_safe(
            X2d_umap, y, classes, cids_all_v2, k,
            v2_fig, "V2 KMeans", "UMAP2", centroids_umap2
        )

        inertias_v2.append(km.inertia_)
        try:
            silhouettes_v2.append(silhouette_score(X_fit_v2, cids_inliers_v2))
        except ValueError:
            silhouettes_v2.append(np.nan)

    plot_metric(k_values, inertias_v2,   "Elbow Method (V2)",      "Inertia",
            os.path.join(v2_fig, "elbow.png"),      color="tab:green", marker="o")
    plot_metric(k_values, silhouettes_v2,"Silhouette Score (V2)", "Silhouette",
            os.path.join(v2_fig, "silhouette.png"), color="tab:red",   marker="s")
    pd.DataFrame({"K": k_values, "Inertia": inertias_v2, "Silhouette": silhouettes_v2}).to_csv(
        os.path.join(v2_out, "metrics.csv"), index=False)
    print("✅ V2 results (PCA100 + UMAP visualizations + robust centroids) saved successfully.")

    print("\nTask 3 complete — both versions executed and saved.")

# -----------------------------
# Main
# -----------------------------
def main():
    out_dir, fig_dir = ensure_dirs(OUT_DIR)

    # Load data
    X, y = load_data(CSV_PATH)
    classes = np.unique(y).tolist()

    # plot_class_counts_from_y(y, out_dir, fig_dir, title="Class Counts", file_stem="class_counts")

    # # --- summaries per model ---
    # knn_res = task1_knn(X, y, classes, out_dir, fig_dir)
    # save_per_class_table("KNN", knn_res.y_true_all, knn_res.y_pred_all, classes, out_dir)

    # svm_results = task2_svm_drop(X, y, classes, out_dir, fig_dir, poly_degree=POLY_DEGREE)
    # for name, res in svm_results.items():
    #     # name is like 'SVM_Linear', 'SVM_Poly', 'SVM_RBF'
    #     save_per_class_table(name, res.y_true_all, res.y_pred_all, classes, out_dir)

    # # Save one CSV with one row per model (KNN + each SVM)
    # save_all_summaries(knn_res, svm_results, out_dir)

    task3_kmeans(X, y, classes, out_dir, fig_dir)

    print("\n All tasks completed successfully. Outputs saved to:", os.path.abspath(out_dir))

if __name__ == "__main__":
    main()
