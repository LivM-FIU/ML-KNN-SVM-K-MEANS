"""
CAP5610 HW4 (DROP version, no CLI): KNN, SVM (Linear/Poly/RBF), KMeans on lncRNA_5_Cancers.csv
- 5-fold Stratified CV
- Macro Accuracy/Precision/Recall/F1, ROC-AUC, PR-AUC
- Confusion matrices and OvR ROC/PR figures
- KMeans for K=2..7 with PCA visualization, Elbow, Silhouette
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    silhouette_score,
)

# -----------------------------
# Config (edit here if needed)
# -----------------------------
RANDOM_STATE = 42
N_SPLITS = 5
CSV_PATH = "data/lncRNA_5_Cancers.csv"  # put your CSV here
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
    Saves:
      - {out_dir}/{file_stem}.csv  (table of counts)
      - {fig_dir}/{file_stem}.png  (bar chart)
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

    # Plot (pure matplotlib; one axes; no explicit colors)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(counts)), counts.values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
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
    Plot OvR ROC and PR curves (per-class) and a macro curve.
    Saves two figures: *_roc.png and *_pr.png
    """
    y_bin = label_binarize(np.array(y_true), classes=classes)
    n_classes = len(classes)

    # ---------- ROC ----------
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro ROC (area via macro mean of TPR over a common FPR grid)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC={roc_auc[i]:.2f})")

    ax.plot(all_fpr, mean_tpr, linestyle="-.", label=f"macro (AUC={roc_auc_macro:.2f})")
    ax.plot([0, 1], [0, 1], linestyle=":")
    ax.set_title(f"{base_name} – ROC (OvR)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{base_name}_roc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- PR ----------
    pr, rc, pr_ap = {}, {}, {}
    for i in range(n_classes):
        pr[i], rc[i], _ = precision_recall_curve(y_bin[:, i], scores[:, i])
        pr_ap[i] = average_precision_score(y_bin[:, i], scores[:, i])  # AP per class

    # Authoritative macro-AP (no plotting needed for this number)
    macro_ap_score = average_precision_score(y_bin, scores, average="macro")

    recall_grid = np.linspace(0, 1, 1000)
    mean_precision = np.zeros_like(recall_grid)
    for i in range(n_classes):
        mean_precision += np.interp(recall_grid, rc[i][::-1], pr[i][::-1])
    mean_precision /= n_classes
    # Area under the macro curve 
    macro_ap_curve = np.trapezoid(mean_precision, recall_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(n_classes):
        ax.plot(rc[i], pr[i], label=f"{classes[i]} (AP={pr_ap[i]:.2f})")
    ax.plot(recall_grid, mean_precision, linestyle="-.", label=f"macro (AP≈{macro_ap_curve:.2f}; score={macro_ap_score:.2f})")
    ax.set_title(f"{base_name} – PR (OvR)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{base_name}_pr.png"), dpi=150, bbox_inches="tight")
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

def task3_kmeans(X: pd.DataFrame, y: pd.Series, classes: List[str], out_dir: str, fig_dir: str):
    print("\n=== Task 3: KMeans Clustering (K=2..7) ===")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d = pca.fit_transform(Xs)

    k_values = list(range(2, 8))
    inertias, silhouettes = [], []
    true_labels = pd.Categorical(y, categories=classes).codes  # 0..C-1

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        cids = km.fit_predict(Xs)

        # PCA visualization: colored by TRUE labels (as assignment requests)
        fig, ax = plt.subplots(figsize=(6, 5))
        for idx, cname in enumerate(classes):
            mask = (true_labels == idx)
            ax.scatter(X2d[mask, 0], X2d[mask, 1], label=cname, s=12, alpha=0.85)
        ax.set_title(f"KMeans Visualization (K={k}) – points colored by true cancer type")
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        ax.legend(fontsize=8, markerscale=1.2)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"task3_k{k}_pca.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, cids))

    # Elbow plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, inertias, marker="o")
    ax.set_title("Elbow Method (Inertia vs K)")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "task3_elbow.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, silhouettes, marker="o")
    ax.set_title("Silhouette Score vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "task3_silhouette.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save numeric results
    pd.DataFrame({"K": k_values, "Inertia": inertias, "Silhouette": silhouettes}).to_csv(
        os.path.join(out_dir, "task3_kmeans_metrics.csv"), index=False
    )

# -----------------------------
# Main
# -----------------------------
def main():
    out_dir, fig_dir = ensure_dirs(OUT_DIR)

    # Load data
    X, y = load_data(CSV_PATH)
    classes = np.unique(y).tolist()

    plot_class_counts_from_y(y, out_dir, fig_dir, title="Class Counts", file_stem="class_counts")

    # --- summaries per model ---
    knn_res = task1_knn(X, y, classes, out_dir, fig_dir)
    svm_results = task2_svm_drop(X, y, classes, out_dir, fig_dir, poly_degree=POLY_DEGREE)
    task3_kmeans(X, y, classes, out_dir, fig_dir)

    # Save one CSV with one row per model (KNN + each SVM)
    save_all_summaries(knn_res, svm_results, out_dir)

    print("\n All tasks completed successfully. Outputs saved to:", os.path.abspath(out_dir))

if __name__ == "__main__":
    main()
