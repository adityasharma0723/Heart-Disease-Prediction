import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def remove_highly_correlated(X, threshold=0.85):
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


def create_composite_features(df):
    df = df.copy()

    if "BMI" in df.columns:
        df["BMI_Category"] = pd.cut(
            df["BMI"],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=["Underweight", "Normal", "Overweight", "Obese"]
        )
        print("[INFO] Created feature: BMI_Category")

    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 35, 55, 100],
            labels=["Young", "Middle", "Senior"]
        )
        print("[INFO] Created feature: Age_Group")

    risk_cols = [
        "Smoking", "Diabetes", "High Blood Pressure",
        "High LDL Cholesterol", "Low HDL Cholesterol",
        "Family Heart Disease"
    ]
    existing_risk_cols = [c for c in risk_cols if c in df.columns]
    if existing_risk_cols:
        risk_df = df[existing_risk_cols].apply(
            lambda col: col.map({"Yes": 1, "No": 0}).fillna(0)
        )
        df["Risk_Score"] = risk_df.sum(axis=1).astype(int)
        print(f"[INFO] Created feature: Risk_Score (from {len(existing_risk_cols)} risk factors)")

    return df


def select_top_features(X, y, feature_names, k=15):
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
        print(f"  - {row['Feature']}: F-Score = {row['F-Score']:.2f}")

    return scores_df, selected_names


def run_feature_engineering(data):
    print("=" * 60)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 60)

    print("\n--- Feature Selection (ANOVA F-test) ---")
    scores_df, selected_names = select_top_features(
        data["X_train"],
        data["y_train"],
        data["feature_names"],
        k=min(15, len(data["feature_names"]))
    )

    data["feature_scores"] = scores_df
    data["selected_features"] = selected_names

    print("\n[Done] Feature engineering complete!\n")
    return data
