import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    target = "Heart Disease Status"
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [FILL] {col}: filled with median = {median_val:.2f}")

    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(df) * 100
            if missing_pct > 20:
                df[col].fillna("Unknown", inplace=True)
                print(f"  [FILL] {col}: {missing_count} missing ({missing_pct:.1f}%) -> filled with 'Unknown'")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  [FILL] {col}: {missing_count} missing -> filled with mode = '{mode_val}'")

    print(f"[INFO] Missing values remaining: {df.isnull().sum().sum()}")
    return df


def encode_features(df):
    target_col = "Heart Disease Status"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.astype(int, errors='ignore')
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"[INFO] Features shape after encoding: {X_encoded.shape}")
    print(f"[INFO] Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return X_encoded, y_encoded, le


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")
    print(f"[INFO] Train class distribution: {np.bincount(y_train)}")
    print(f"[INFO] Test class distribution:  {np.bincount(y_test)}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    print(f"[INFO] Before SMOTE - Class distribution: {np.bincount(y_train)}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE  - Class distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


def run_preprocessing_pipeline(filepath):
    print("=" * 60)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    print("\n--- Loading Data ---")
    df = load_data(filepath)

    print("\n--- Handling Missing Values ---")
    df = handle_missing_values(df)

    print("\n--- Creating Composite Features ---")
    from src.feature_engineering import create_composite_features
    df = create_composite_features(df)

    print("\n--- Encoding Features ---")
    X, y, label_encoder = encode_features(df)

    print("\n--- Removing Highly Correlated Features ---")
    from src.feature_engineering import remove_highly_correlated
    X, dropped_cols = remove_highly_correlated(X)

    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(X, y)

    feature_names = X.columns.tolist()

    print("\n--- Scaling Features ---")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n--- Applying SMOTE ---")
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    print("\n[Done] Preprocessing complete!\n")

    return {
        "df_raw": df,
        "X_train": X_train_resampled,
        "X_test": X_test_scaled,
        "y_train": y_train_resampled,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "X_train_original": X_train_scaled,
        "y_train_original": y_train,
    }
