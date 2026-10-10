"""Evaluation helpers: metrics, comparison tables, confusion matrices, ROC."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["No Diabetes", "Diabetes"]


def evaluate_model(model, X_test, y_test, name):
    """Compute the main classification metrics for one fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Specificity": tn / (tn + fp),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def evaluate_models(models, X_test, y_test):
    """Evaluate all models; return a DataFrame sorted by ROC-AUC."""
    rows = [evaluate_model(m, X_test, y_test, name) for name, m in models.items()]
    return (
        pd.DataFrame(rows)
        .sort_values("ROC-AUC", ascending=False)
        .reset_index(drop=True)
    )


def print_classification_reports(models, X_test, y_test):
    for name, model in models.items():
        print("=" * 60)
        print(name)
        print("=" * 60)
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=CLASS_NAMES, digits=3))


def plot_confusion_matrices(models, X_test, y_test, save_path=None):
    """One labelled confusion matrix per model in a single figure."""
    n = len(models)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.3 * rows))
    axes = axes.ravel()
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            annot_kws={"size": 14},
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
        )
        ax.tick_params(axis="y", rotation=0)
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices on the Test Set", fontsize=15)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(models, X_test, y_test, save_path=None):
    """ROC curves of all models on one axis, with AUC in the legend."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random guess (AUC = 0.500)")
    ax.set_title("ROC Curves - Test Set")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def best_per_metric(results, metrics=("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")):
    """Which model achieves the highest value of each metric."""
    rows = []
    for metric in metrics:
        idx = results[metric].idxmax()
        rows.append({"Metric": metric,
                     "Best model": results.loc[idx, "Model"],
                     "Value": round(results.loc[idx, metric], 4)})
    return pd.DataFrame(rows)
