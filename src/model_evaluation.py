"""
model_evaluation.py
-------------------
Comprehensive model evaluation for the Heart Disease Prediction pipeline.

For each model, this module generates:
  - Accuracy, Precision, Recall, F1-Score
  - Classification Report
  - Confusion Matrix (heatmap)
  - ROC-AUC Curve
  - Model Comparison Table
  - Feature Importance (for tree-based models)
  - SHAP explanations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    ConfusionMatrixDisplay
)


# ---------------------------------------------------------------------------
#  1. EVALUATE A SINGLE MODEL
# ---------------------------------------------------------------------------

def evaluate_single_model(name, y_true, y_pred, class_names):
    """
    Calculate all metrics for a single model.

    Parameters
    ----------
    name : str
        Model name.
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_names : list
        Class label names.

    Returns
    -------
    dict
        Dictionary of metric values.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    }

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return metrics


# ---------------------------------------------------------------------------
#  2. CONFUSION MATRIX PLOT
# ---------------------------------------------------------------------------

def plot_confusion_matrices(predictions, y_test, class_names, output_dir):
    """
    Plot confusion matrix heatmaps for all models.
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filepath}")


# ---------------------------------------------------------------------------
#  3. ROC-AUC CURVES
# ---------------------------------------------------------------------------

def plot_roc_curves(probabilities, y_test, output_dir):
    """
    Plot ROC curves for all models overlaid on a single plot.
    """
    plt.figure(figsize=(8, 6))

    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filepath}")


# ---------------------------------------------------------------------------
#  4. MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------

def create_comparison_table(all_metrics, probabilities, y_test, output_dir):
    """
    Create a comparison table of all model metrics.
    """
    df = pd.DataFrame(all_metrics)

    # Add ROC-AUC
    roc_aucs = []
    for row in all_metrics:
        name = row["Model"]
        if name in probabilities:
            roc_aucs.append(round(roc_auc_score(y_test, probabilities[name]), 4))
        else:
            roc_aucs.append("N/A")
    df["ROC-AUC"] = roc_aucs

    # Sort by F1-Score descending
    df = df.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    # Identify best model
    best_model = df.iloc[0]["Model"]
    best_f1 = df.iloc[0]["F1-Score"]
    print(f"\n  🏆 Best Model: {best_model} (F1-Score = {best_f1})")

    # Save table
    filepath = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(filepath, index=False)
    print(f"[SAVED] {filepath}")

    return df, best_model


# ---------------------------------------------------------------------------
#  5. FEATURE IMPORTANCE PLOT
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names, output_dir, top_n=15):
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("[WARN] Model does not have feature_importances_. Skipping.")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances", fontsize=14, fontweight="bold")
    plt.barh(
        range(len(indices)),
        importances[indices][::-1],
        color="steelblue",
        edgecolor="navy"
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filepath}")

    # Return as DataFrame
    fi_df = pd.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": importances[indices]
    })
    return fi_df


# ---------------------------------------------------------------------------
#  6. SHAP EXPLANATIONS
# ---------------------------------------------------------------------------

def generate_shap_explanations(model, X_test, feature_names, output_dir):
    """
    Generate SHAP summary plot for model interpretability.

    SHAP (SHapley Additive exPlanations) shows:
    - Which features have the biggest impact on predictions
    - Whether high/low feature values push toward positive/negative prediction
    """
    try:
        import shap

        print("\n--- Generating SHAP Explanations ---")

        # Use TreeExplainer for tree-based models, KernelExplainer for others
        model_type = type(model).__name__
        if model_type in ["RandomForestClassifier", "DecisionTreeClassifier"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # For binary classification, take class 1 (heart disease = Yes)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Use a small background sample for KernelExplainer
            background = shap.sample(X_test, min(100, len(X_test)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test[:100])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            X_test = X_test[:100]

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8)
        )
        plt.title("SHAP Feature Impact Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {filepath}")

    except ImportError:
        print("[WARN] SHAP library not installed. Run: pip install shap")
    except Exception as e:
        print(f"[WARN] SHAP generation failed: {e}")


# ---------------------------------------------------------------------------
#  7. FULL EVALUATION PIPELINE
# ---------------------------------------------------------------------------

def run_model_evaluation(data: dict, output_dir: str = "outputs") -> dict:
    """
    Execute the full evaluation pipeline.

    Parameters
    ----------
    data : dict
        Pipeline data containing models, predictions, etc.
    output_dir : str
        Directory to save plots and reports.

    Returns
    -------
    dict
        Updated data with evaluation results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  STEP 4: MODEL EVALUATION")
    print("=" * 60)

    y_test = data["y_test"]
    predictions = data["predictions"]
    probabilities = data["probabilities"]
    class_names = list(data["label_encoder"].classes_)

    # Evaluate each model
    print("\n--- Per-Model Metrics ---")
    all_metrics = []
    for name, y_pred in predictions.items():
        metrics = evaluate_single_model(name, y_test, y_pred, class_names)
        all_metrics.append(metrics)

    # Confusion matrices
    print("\n--- Confusion Matrices ---")
    plot_confusion_matrices(predictions, y_test, class_names, output_dir)

    # ROC curves
    print("\n--- ROC-AUC Curves ---")
    plot_roc_curves(probabilities, y_test, output_dir)

    # Comparison table
    comparison_df, best_model_name = create_comparison_table(
        all_metrics, probabilities, y_test, output_dir
    )

    # Feature importance (Best model)
    best_model = data["trained_models"][best_model_name]
    print("\n--- Feature Importance ---")
    fi_df = plot_feature_importance(
        best_model, data["feature_names"], output_dir
    )

    # SHAP
    generate_shap_explanations(
        best_model, data["X_test"], data["feature_names"], output_dir
    )

    data["comparison_df"] = comparison_df
    data["best_model_name"] = best_model_name
    data["best_model"] = best_model
    data["feature_importance"] = fi_df

    print("\n[✓] Model evaluation complete!\n")
    return data
