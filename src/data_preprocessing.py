"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, encoding, scaling, splitting,
and class imbalance treatment for the Heart Disease Prediction pipeline.

Why each step matters:
  - Missing values: Models can't handle NaN — we impute smartly.
  - Encoding: ML models need numeric inputs.
  - Scaling: Distance-based models (KNN, SVM) and regularized models
    (Logistic Regression) require features on the same scale.
  - SMOTE: Our dataset has 80/20 class imbalance. Without balancing,
    models learn to always predict the majority class ("No").
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
#  1. LOAD DATA
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the heart disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
#  2. HANDLE MISSING VALUES
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the dataset.

    Strategy:
      - Numeric columns → median (robust to outliers)
      - Categorical columns → mode (most frequent value)
      - 'Alcohol Consumption' has ~25% missing → use 'Unknown' category

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potential missing values.

    Returns
    -------
    pd.DataFrame
        Dataframe with no missing values.
    """
    df = df.copy()

    # Separate column types
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from lists if present
    target = "Heart Disease Status"
    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    # Fill numeric columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [FILL] {col}: {df[col].isnull().sum()} → filled with median = {median_val:.2f}")

    # Fill categorical columns
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_pct = missing_count / len(df) * 100
            if missing_pct > 20:
                # Too many missing — create an 'Unknown' category
                df[col].fillna("Unknown", inplace=True)
                print(f"  [FILL] {col}: {missing_count} missing ({missing_pct:.1f}%) → filled with 'Unknown'")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  [FILL] {col}: {missing_count} missing → filled with mode = '{mode_val}'")

    print(f"[INFO] Missing values remaining: {df.isnull().sum().sum()}")
    return df


# ---------------------------------------------------------------------------
#  3. ENCODE FEATURES
# ---------------------------------------------------------------------------

def encode_features(df: pd.DataFrame):
    """
    Encode categorical features and the target variable.

    - Features: One-hot encoding with drop_first=True
      (avoids multicollinearity / dummy variable trap)
    - Target: LabelEncoder (No → 0, Yes → 1)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    tuple
        (X_encoded, y_encoded, label_encoder)
    """
    target_col = "Heart Disease Status"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Ensure all columns are numeric (bool → int)
    X_encoded = X_encoded.astype(int, errors='ignore')

    # Label encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"[INFO] Features shape after encoding: {X_encoded.shape}")
    print(f"[INFO] Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X_encoded, y_encoded, le


# ---------------------------------------------------------------------------
#  4. SPLIT DATA
# ---------------------------------------------------------------------------

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split into training and test sets with stratification.

    Stratification ensures both sets maintain the original class ratio,
    which is critical for imbalanced datasets.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    test_size : float
        Fraction for test set.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # maintain class proportions
    )

    print(f"[INFO] Train set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set:  {X_test.shape[0]} samples")
    print(f"[INFO] Train class distribution: {np.bincount(y_train)}")
    print(f"[INFO] Test class distribution:  {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
#  5. SCALE FEATURES
# ---------------------------------------------------------------------------

def scale_features(X_train, X_test):
    """
    Standardize features to zero mean and unit variance.

    The scaler is fitted ONLY on training data to prevent data leakage.

    Parameters
    ----------
    X_train : array-like
    X_test : array-like

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[INFO] Features scaled using StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
#  6. HANDLE CLASS IMBALANCE WITH SMOTE
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    SMOTE creates synthetic samples of the minority class by interpolating
    between existing minority samples. Applied ONLY to training data.

    Parameters
    ----------
    X_train : array-like
    y_train : array-like
    random_state : int

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    print(f"[INFO] Before SMOTE — Class distribution: {np.bincount(y_train)}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"[INFO] After SMOTE  — Class distribution: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
#  7. FULL PIPELINE
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(filepath: str):
    """
    Execute the full data preprocessing pipeline.

    Steps:
      1. Load data
      2. Handle missing values
      3. Encode features & target
      4. Train-test split (stratified)
      5. Scale features
      6. Apply SMOTE on training data

    Returns
    -------
    dict
        Dictionary containing all processed data and fitted objects.
    """
    print("=" * 60)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # 1. Load
    print("\n--- Loading Data ---")
    df = load_data(filepath)

    # 2. Clean
    print("\n--- Handling Missing Values ---")
    df = handle_missing_values(df)

    # 3. Encode
    print("\n--- Encoding Features ---")
    X, y, label_encoder = encode_features(df)

    # 4. Split
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save column names before scaling
    feature_names = X.columns.tolist()

    # 5. Scale
    print("\n--- Scaling Features ---")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 6. SMOTE
    print("\n--- Applying SMOTE ---")
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    print("\n[✓] Preprocessing complete!\n")

    return {
        "df_raw": df,
        "X_train": X_train_resampled,
        "X_test": X_test_scaled,
        "y_train": y_train_resampled,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        # Keep originals for EDA / comparison
        "X_train_original": X_train_scaled,
        "y_train_original": y_train,
    }
