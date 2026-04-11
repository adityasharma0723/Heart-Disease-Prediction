"""
hyperparameter_tuning.py
------------------------
Optimize the best-performing model using GridSearchCV.

Why GridSearchCV?
  - Systematically tries all combinations of hyperparameters
  - Uses cross-validation to prevent overfitting to a single train/test split
  - We use F1 scoring (not accuracy) because of class imbalance

Strategy:
  - Identify the best model from evaluation
  - Define model-specific hyperparameter grids
  - Run GridSearchCV with 5-fold stratified cross-validation
  - Compare before vs after performance
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)


# ---------------------------------------------------------------------------
#  HYPERPARAMETER GRIDS FOR EACH MODEL
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"]
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000],
        "class_weight": ["balanced"]
    },
    "Decision Tree": {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced"]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced"]
    }
}


# ---------------------------------------------------------------------------
#  TUNE MODEL
# ---------------------------------------------------------------------------

def tune_model(model, model_name, X_train, y_train, cv=5, scoring="f1"):
    """
    Run GridSearchCV on the specified model.

    Parameters
    ----------
    model : estimator
        The sklearn model instance.
    model_name : str
        Name of the model (used to look up param grid).
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Optimization metric.

    Returns
    -------
    tuple
        (best_model, best_params, best_score)
    """
    if model_name not in PARAM_GRIDS:
        print(f"[WARN] No param grid defined for '{model_name}'. Skipping tuning.")
        return model, {}, 0.0

    param_grid = PARAM_GRIDS[model_name]
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"  Grid size: {total_combos} combinations × {cv} folds = {total_combos * cv} fits")

    # Use RandomizedSearchCV if grid is too large
    if total_combos > 200:
        from sklearn.model_selection import RandomizedSearchCV
        print("  [INFO] Grid too large — using RandomizedSearchCV (100 iterations)")
        search = RandomizedSearchCV(
            model, param_grid,
            n_iter=100,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
    else:
        search = GridSearchCV(
            model, param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

    search.fit(X_train, y_train)

    print(f"  Best Score (CV): {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


# ---------------------------------------------------------------------------
#  COMPARE BEFORE / AFTER
# ---------------------------------------------------------------------------

def compare_before_after(model_before, model_after, X_test, y_test, model_name, output_dir):
    """
    Show improvement table comparing original vs tuned model.
    """
    results = []
    for label, model in [("Before Tuning", model_before), ("After Tuning", model_after)]:
        y_pred = model.predict(X_test)
        results.append({
            "Stage": label,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

    df = pd.DataFrame(results)
    print(f"\n{'─' * 60}")
    print(f"  HYPERPARAMETER TUNING RESULTS: {model_name}")
    print(f"{'─' * 60}")
    print(df.to_string(index=False))

    # Calculate improvement
    improvement = results[1]["F1-Score"] - results[0]["F1-Score"]
    print(f"\n  F1-Score Improvement: {improvement:+.4f}")

    filepath = os.path.join(output_dir, "tuning_comparison.csv")
    df.to_csv(filepath, index=False)
    print(f"[SAVED] {filepath}")

    return df


# ---------------------------------------------------------------------------
#  RUN HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

def run_hyperparameter_tuning(data: dict, output_dir: str = "outputs") -> dict:
    """
    Execute hyperparameter tuning on the best model.

    Parameters
    ----------
    data : dict
        Pipeline data.
    output_dir : str
        Directory to save results.

    Returns
    -------
    dict
        Updated data with tuned model.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  STEP 5: HYPERPARAMETER TUNING")
    print("=" * 60)

    best_model_name = data["best_model_name"]
    best_model_original = data["best_model"]

    print(f"\n  Tuning: {best_model_name}")
    print(f"  Scoring: F1-Score (optimized for class imbalance)\n")

    # Clone the model to preserve original
    from sklearn.base import clone
    model_to_tune = clone(best_model_original)

    # Run tuning
    tuned_model, best_params, best_cv_score = tune_model(
        model_to_tune, best_model_name,
        data["X_train"], data["y_train"]
    )

    # Compare
    compare_before_after(
        best_model_original, tuned_model,
        data["X_test"], data["y_test"],
        best_model_name, output_dir
    )

    # Update data
    data["tuned_model"] = tuned_model
    data["tuned_params"] = best_params
    data["tuned_predictions"] = tuned_model.predict(data["X_test"])

    print("\n[✓] Hyperparameter tuning complete!\n")
    return data
