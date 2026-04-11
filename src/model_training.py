"""
model_training.py
-----------------
Train multiple classification models for Heart Disease Prediction.

Models trained:
  1. Logistic Regression  — simple, interpretable linear model
  2. Decision Tree         — non-linear, easy to visualize
  3. Random Forest         — ensemble of decision trees (reduces overfitting)
  4. SVM (Support Vector Machine) — powerful for non-linear boundaries

All models use class_weight='balanced' to handle class imbalance,
in addition to SMOTE resampling applied during preprocessing.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def build_models(random_state=42):
    """
    Create a dictionary of ML models with their configurations.

    Using class_weight='balanced' makes the model assign higher penalties
    for misclassifying the minority class (heart disease = Yes).

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
            max_depth=10
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        ),
        "SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,       # needed for ROC-AUC
            random_state=random_state
        ),
    }

    return models


def train_models(models: dict, X_train, y_train):
    """
    Train all models on the training set.

    Parameters
    ----------
    models : dict
        {model_name: model_instance}
    X_train : array-like
        Training features (scaled + resampled).
    y_train : array-like
        Training target.

    Returns
    -------
    dict
        {model_name: trained_model}
    """
    trained = {}

    for name, model in models.items():
        print(f"  Training {name}...", end=" ")
        model.fit(X_train, y_train)
        trained[name] = model
        print("Done ✓")

    return trained


def run_model_training(data: dict) -> dict:
    """
    Execute the model training pipeline.

    Parameters
    ----------
    data : dict
        Output from preprocessing/feature engineering pipeline.

    Returns
    -------
    dict
        Updated data dict with trained models and predictions.
    """
    print("=" * 60)
    print("  STEP 3: MODEL BUILDING")
    print("=" * 60)

    # Build models
    print("\n--- Building Models ---")
    models = build_models()

    # Train
    print("\n--- Training Models ---")
    trained_models = train_models(models, data["X_train"], data["y_train"])

    # Generate predictions
    print("\n--- Generating Predictions ---")
    predictions = {}
    probabilities = {}
    for name, model in trained_models.items():
        predictions[name] = model.predict(data["X_test"])
        # Get probability scores for ROC-AUC
        if hasattr(model, "predict_proba"):
            probabilities[name] = model.predict_proba(data["X_test"])[:, 1]
        elif hasattr(model, "decision_function"):
            probabilities[name] = model.decision_function(data["X_test"])
        print(f"  Predictions generated for {name} ✓")

    data["trained_models"] = trained_models
    data["predictions"] = predictions
    data["probabilities"] = probabilities

    print("\n[✓] Model training complete!\n")
    return data
