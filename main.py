"""
main.py — Heart Disease Prediction Pipeline
=============================================
End-to-end ML pipeline that executes all steps:

  1. Data Preprocessing  (load, clean, encode, scale, SMOTE)
  2. Feature Engineering (feature selection & analysis)
  3. Model Training      (Logistic Regression, Decision Tree, Random Forest, SVM)
  4. Model Evaluation    (metrics, confusion matrix, ROC-AUC, SHAP)
  5. Hyperparameter Tuning (GridSearchCV on best model)
  6. Save Model          (joblib serialization)

Usage:
    python main.py

Author: Aditya Sharma
"""

import os
import sys
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import run_preprocessing_pipeline
from src.feature_engineering import run_feature_engineering
from src.model_training import run_model_training
from src.model_evaluation import run_model_evaluation
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.predict import save_model_artifacts


def main():
    """Run the full ML pipeline."""
    start_time = time.time()

    print("\n" + "+" + "=" * 58 + "+")
    print("|  HEART DISEASE PREDICTION - ML PIPELINE                  |")
    print("+" + "=" * 58 + "+\n")

    # Paths
    data_path = os.path.join(PROJECT_ROOT, "data", "heart_disease.csv")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    model_dir = os.path.join(PROJECT_ROOT, "models")

    # ─── Step 1: Data Preprocessing ───
    data = run_preprocessing_pipeline(data_path)

    # ─── Step 2: Feature Engineering ───
    data = run_feature_engineering(data)

    # ─── Step 3: Model Training ───
    data = run_model_training(data)

    # ─── Step 4: Model Evaluation ───
    data = run_model_evaluation(data, output_dir=output_dir)

    # ─── Step 5: Hyperparameter Tuning ───
    data = run_hyperparameter_tuning(data, output_dir=output_dir)

    # ─── Step 6: Save Model ───
    print("=" * 60)
    print("  STEP 6: SAVING MODEL")
    print("=" * 60)
    save_model_artifacts(data, output_dir=model_dir)

    # ─── Summary ───
    elapsed = time.time() - start_time
    print("\n" + "+" + "=" * 58 + "+")
    print("|  PIPELINE COMPLETE                                       |")
    print("+" + "=" * 58 + "+")
    print(f"\n  Total time: {elapsed:.1f} seconds")
    print(f"  Best model: {data['best_model_name']}")
    print(f"  Outputs saved to: {output_dir}/")
    print(f"  Model saved to:   {model_dir}/")
    print(f"\n  Run predictions with:")
    print(f"    python src/predict.py\n")


if __name__ == "__main__":
    main()
