"""
feature_engineering.py
----------------------
Feature selection and creation for the Heart Disease Prediction pipeline.

This module:
  - Identifies and removes highly correlated redundant features
  - Creates new composite features from domain knowledge
  - Selects top features using statistical tests
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


# ---------------------------------------------------------------------------
#  1. REMOVE HIGHLY CORRELATED FEATURES
# ---------------------------------------------------------------------------

def remove_highly_correlated(X: pd.DataFrame, threshold: float = 0.85):
    """
    Remove one of each pair of features with correlation > threshold.

    Why: Highly correlated features carry redundant information and can
    hurt model performance (multicollinearity).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Correlation threshold above which one feature is dropped.

    Returns
    -------
    tuple
        (filtered_df, list_of_dropped_columns)
    """
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > threshold)
    ]

    if to_drop:
        print(f"[INFO] Dropping {len(to_drop)} highly correlated features: {to_drop}")
    else:
        print("[INFO] No highly correlated feature pairs found (threshold={:.2f})".format(threshold))

    return X.drop(columns=to_drop), to_drop


# ---------------------------------------------------------------------------
#  2. CREATE NEW FEATURES
# ---------------------------------------------------------------------------

def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features from domain knowledge.

    New features:
      - BMI_Category: Categorize BMI into health brackets
      - Age_Group: Categorize age into life stages
      - Risk_Score: Composite score from key risk factors

    Parameters
    ----------
    df : pd.DataFrame
        Raw / cleaned dataframe (before encoding).

    Returns
    -------
    pd.DataFrame
        Dataframe with new columns added.
    """
    df = df.copy()

    # --- BMI Category ---
    # WHO BMI brackets: <18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, 30+ obese
    if "BMI" in df.columns:
        df["BMI_Category"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        )
        print("[INFO] Created feature: BMI_Category")

    # --- Age Group ---
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 35, 55, 100],
            labels=["Young", "Middle", "Senior"]
        )
        print("[INFO] Created feature: Age_Group")

    # --- Risk Score ---
    # Combine binary risk factors into a single count
    risk_cols = [
        "Smoking", "Diabetes", "High Blood Pressure",
        "High LDL Cholesterol", "Low HDL Cholesterol",
        "Family Heart Disease"
    ]
    existing_risk_cols = [c for c in risk_cols if c in df.columns]
    if existing_risk_cols:
        # Convert Yes/No to 1/0 and sum
        risk_df = df[existing_risk_cols].apply(
            lambda col: col.map({"Yes": 1, "No": 0}).fillna(0)
        )
        df["Risk_Score"] = risk_df.sum(axis=1).astype(int)
        print(f"[INFO] Created feature: Risk_Score (from {len(existing_risk_cols)} risk factors)")

    return df


# ---------------------------------------------------------------------------
#  3. SELECT TOP FEATURES
# ---------------------------------------------------------------------------

def select_top_features(X, y, feature_names, k=15):
    """
    Select top-k features using ANOVA F-test (SelectKBest).

    Why: Removing irrelevant features reduces noise and improves
    model generalization.

    Parameters
    ----------
    X : array-like
        Scaled feature matrix.
    y : array-like
        Target vector.
    feature_names : list
        Column names corresponding to X.
    k : int
        Number of top features to select.

    Returns
    -------
    tuple
        (feature_scores_df, selected_feature_names)
    """
    # Ensure k doesn't exceed number of features
    k = min(k, X.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "F-Score": selector.scores_,
        "p-value": selector.pvalues_
    }).sort_values("F-Score", ascending=False)

    selected_mask = selector.get_support()
    selected_names = [f for f, m in zip(feature_names, selected_mask) if m]

    print(f"\n[INFO] Top {k} features selected:")
    for i, row in scores_df.head(k).iterrows():
        print(f"  • {row['Feature']}: F-Score = {row['F-Score']:.2f}")

    return scores_df, selected_names


# ---------------------------------------------------------------------------
#  4. RUN FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def run_feature_engineering(data: dict) -> dict:
    """
    Execute the feature engineering pipeline.

    Note: For this project, we keep all features after one-hot encoding
    and let the models + SelectKBest determine importance.
    The composite features are added BEFORE encoding in the preprocessing step.

    Parameters
    ----------
    data : dict
        Output from preprocessing pipeline.

    Returns
    -------
    dict
        Updated data dictionary with feature scores.
    """
    print("=" * 60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    # Feature importance analysis using SelectKBest
    print("\n--- Feature Selection (ANOVA F-test) ---")
    scores_df, selected_names = select_top_features(
        data["X_train"],
        data["y_train"],
        data["feature_names"],
        k=min(15, len(data["feature_names"]))
    )

    data["feature_scores"] = scores_df
    data["selected_features"] = selected_names

    print("\n[✓] Feature engineering complete!\n")
    return data
