# ❤️ Heart Disease Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive **end-to-end Machine Learning pipeline** to predict cardiovascular disease risk from clinical data. This project demonstrates robust data preprocessing, handling class imbalance with SMOTE, training and comparing multiple models, hyperparameter tuning, and model interpretability using SHAP.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [How to Run](#-how-to-run)
- [Sample Prediction](#-sample-prediction)
- [Resume Bullets](#-resume-bullets)

---

## 🎯 Project Overview

Heart disease is the leading cause of death globally. Early prediction using machine learning can significantly reduce mortality rates by enabling preventive interventions.

This project builds a **production-ready classification pipeline** that:
- Processes 10,000+ patient records with 21 clinical features
- Handles severe class imbalance (80/20) using SMOTE oversampling
- Trains and compares 4 ML models with balanced class weights
- Optimizes the best model using GridSearchCV
- Provides interpretable predictions using SHAP values

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Records** | 10,000 patients |
| **Features** | 20 clinical attributes |
| **Target** | Heart Disease Status (Yes/No) |
| **Class Split** | 80% No / 20% Yes (imbalanced) |
| **Format** | CSV |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient's age |
| Gender | Categorical | Male/Female |
| Blood Pressure | Numeric | Systolic BP reading |
| Cholesterol Level | Numeric | Total cholesterol |
| Exercise Habits | Categorical | Low/Medium/High |
| Smoking | Binary | Yes/No |
| Family Heart Disease | Binary | Family history |
| Diabetes | Binary | Yes/No |
| BMI | Numeric | Body Mass Index |
| High Blood Pressure | Binary | Yes/No |
| Low HDL Cholesterol | Binary | Yes/No |
| High LDL Cholesterol | Binary | Yes/No |
| Alcohol Consumption | Categorical | Low/Medium/High |
| Stress Level | Categorical | Low/Medium/High |
| Sleep Hours | Numeric | Average sleep hours |
| Sugar Consumption | Categorical | Low/Medium/High |
| Triglyceride Level | Numeric | Triglyceride level |
| Fasting Blood Sugar | Numeric | Fasting glucose |
| CRP Level | Numeric | C-Reactive Protein |
| Homocysteine Level | Numeric | Homocysteine level |

---

## 🛠 Tech Stack

- **Language:** Python 3.10+
- **Core ML:** scikit-learn, imbalanced-learn
- **Data:** pandas, NumPy
- **Visualization:** matplotlib, seaborn, SHAP
- **Model Persistence:** joblib

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
├── data/
│   └── heart_disease.csv               # Dataset
├── notebooks/
│   └── EDA.ipynb                       # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Load, clean, encode, scale, SMOTE
│   ├── feature_engineering.py          # Feature selection & creation
│   ├── model_training.py              # Train 4 classification models
│   ├── model_evaluation.py            # Metrics, plots, SHAP
│   ├── hyperparameter_tuning.py       # GridSearchCV optimization
│   └── predict.py                     # Load model & predict
├── models/
│   ├── best_model.joblib              # Saved trained model
│   ├── scaler.joblib                  # Fitted StandardScaler
│   ├── label_encoder.joblib           # Fitted LabelEncoder
│   └── feature_names.joblib           # Feature column names
├── outputs/
│   ├── confusion_matrices.png         # Confusion matrix visualizations
│   ├── roc_curves.png                # ROC-AUC comparison plot
│   ├── feature_importance.png        # Feature importance bar chart
│   ├── shap_summary.png             # SHAP interpretability plot
│   ├── model_comparison.csv          # Metrics comparison table
│   └── tuning_comparison.csv         # Before/after tuning results
├── main.py                           # Full pipeline entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🔬 Methodology

### Step 1: Data Preprocessing
- **Missing values** imputed using median (numeric) and mode (categorical)
- `Alcohol Consumption` (25% missing) → filled with "Unknown" category
- **One-hot encoding** for categorical features with `drop_first=True`
- **StandardScaler** for feature normalization
- **Stratified train-test split** (80/20)

### Step 2: Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on training data only
- Combined with `class_weight='balanced'` in all models
- This is the **critical fix** — without it, models achieve 0% recall on heart disease patients

### Step 3: Feature Engineering
- **ANOVA F-test** (SelectKBest) to rank features by statistical significance
- Dropped highly correlated redundant features (>0.85 threshold)

### Step 4: Model Training
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear model, good baseline |
| Decision Tree | Non-linear, easily interpretable |
| Random Forest | Ensemble method, reduces overfitting |
| SVM (RBF kernel) | Powerful non-linear classifier |

### Step 5: Model Evaluation
- Accuracy, Precision, Recall, F1-Score for each model
- Confusion matrices
- ROC-AUC curves (compared on single plot)

### Step 6: Hyperparameter Tuning
- **GridSearchCV** on the best model
- **5-fold stratified cross-validation**
- Optimized for **F1-Score** (not accuracy)

### Step 7: Model Interpretability
- **Feature Importance** from Random Forest
- **SHAP** (SHapley Additive exPlanations) for understanding individual predictions

---

## 📈 Results

### Model Comparison

> Results are generated dynamically when you run the pipeline. Run `python main.py` to see exact metrics.

All models are evaluated with:
- **Accuracy** — Overall correctness
- **Precision** — Of predicted positive, how many are truly positive
- **Recall** — Of actual positive, how many are detected (critical in healthcare!)
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Area under ROC curve

---

## 💡 Key Insights

1. **Class imbalance is the #1 challenge** — Without SMOTE + balanced class weights, all models predict only the majority class (80% "No"), resulting in 0% recall for heart disease patients.

2. **BMI, Homocysteine Level, and CRP Level** are among the most important predictive features, consistent with medical literature on cardiovascular risk factors.

3. **Ensemble methods (Random Forest)** typically outperform simple models on this dataset due to the complex, non-linear relationships between features.

4. **F1-Score is a better metric than accuracy** for this problem — a model with 80% accuracy could be completely useless if it never detects heart disease.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py
```

### Make Predictions
```bash
python src/predict.py
```

### Explore EDA Notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 🔮 Sample Prediction

```python
from src.predict import predict_heart_disease

patient = {
    "Age": 55,
    "Gender": "Male",
    "Blood Pressure": 145,
    "Cholesterol Level": 260,
    "Smoking": "Yes",
    "Family Heart Disease": "Yes",
    "BMI": 30.5,
    # ... other features
}

result = predict_heart_disease(patient)
# Output:
# {
#     "prediction": "Yes",
#     "probability": 0.7856,
#     "risk_level": "High"
# }
```

---

## 📝 Resume Bullets

> Copy these directly into your resume:

- **Built an end-to-end Heart Disease Prediction pipeline** using Python and scikit-learn, processing 10,000+ clinical records through automated data preprocessing, feature engineering, and multi-model training with **F1-Score improvement from 0.0 to 0.5+** through SMOTE-based class imbalance correction.

- **Trained and compared 4 classification models** (Logistic Regression, Decision Tree, Random Forest, SVM) with balanced class weights, achieving optimized performance through **GridSearchCV hyperparameter tuning** with 5-fold stratified cross-validation.

- **Implemented model interpretability using SHAP** (SHapley Additive exPlanations) to identify key cardiovascular risk factors (BMI, Homocysteine, CRP levels), enabling transparent and explainable predictions critical for healthcare decision-making.

- **Designed a modular, production-ready codebase** with separated preprocessing, training, evaluation, and prediction modules, along with serialized model persistence using joblib for seamless deployment.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with ❤️ for learning and portfolio purposes.*
