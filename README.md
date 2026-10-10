# Diabetes Onset Prediction Using Machine Learning

An end-to-end Predictive Analytics / KDD project that predicts whether a patient is likely to have diabetes from routine diagnostic measurements, using the Pima Indians Diabetes Dataset.

| | |
|---|---|
| **Student** | Hardik Jain |
| **Student ID** | 23070521056 |
| **Course** | Predictive Analytics |
| **Project type** | End-to-End Predictive Analytics Model (KDD process) |

> **Disclaimer:** This project is for academic and research purposes only. The model is **not** a medical diagnostic system and must not be used for clinical decisions.

### Deliverables

| Deliverable | File |
|---|---|
| Jupyter Notebook (fully executed) | [`notebooks/Diabetes_Onset_Prediction.ipynb`](notebooks/Diabetes_Onset_Prediction.ipynb) |
| Project Report | [`reports/Diabetes_Onset_Prediction_Report.pdf`](reports/Diabetes_Onset_Prediction_Report.pdf) |
| Presentation (12 slides + speaker notes) | [`presentation/Diabetes_Onset_Prediction_Presentation.md`](presentation/Diabetes_Onset_Prediction_Presentation.md) |
| Results | [`results/model_comparison.csv`](results/model_comparison.csv), [`results/metrics.csv`](results/metrics.csv), [`results/figures/`](results/figures) |

---

## Problem Statement

Type 2 diabetes often develops slowly and can go unnoticed for years. Given a patient's diagnostic measurements **X** (glucose, BMI, blood pressure, insulin, age, pregnancies, skin-fold thickness and diabetes pedigree function), predict whether **Outcome = 1 (diabetes)** or **Outcome = 0 (no diabetes)**, and estimate the probability of diabetes.

## Objective

* Build a leak-free, reproducible machine-learning pipeline that walks through every KDD stage.
* Compare five classification algorithms using metrics suited to an imbalanced screening problem.
* Interpret the selected model with feature importance and SHAP.
* Show how such a model could *support* early screening by flagging higher-risk individuals for proper clinical testing.

## Dataset

**Pima Indians Diabetes Dataset.** It was originally collected by the US National Institute of Diabetes and Digestive and Kidney Diseases and is distributed via the UCI Machine Learning Repository.

* Source used: <https://github.com/jbrownlee/Datasets/blob/master/pima-indians-diabetes.data.csv>
* Stored at: `data/pima-indians-diabetes.csv`. This is the original 768 observations with a header row added. No values were modified.
* 768 patients (females of Pima Indian heritage, aged at least 21), 8 features, 1 binary target.
* Class balance: 500 non-diabetic (65.1 %) and 268 diabetic (34.9 %).

If the CSV is missing, the notebook tries to download it automatically. Without internet access, download the file from the link above, add the header row below and save it as `data/pima-indians-diabetes.csv`:

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

## Features

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration, 2 h oral glucose tolerance test (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/mL) |
| BMI | Body-mass index (kg/m²) |
| DiabetesPedigreeFunction | Score of diabetes likelihood based on family history |
| Age | Age in years |
| **Outcome** | **Target: 0 = no diabetes, 1 = diabetes** |

## Methodology

1. **Data cleaning:** zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically impossible and are treated as missing (NaN). Insulin is 48.7 % missing and SkinThickness 29.6 %.
2. **Stratified 80/20 train-test split** (`random_state=42`) *before* any fitting: 614 training and 154 test patients.
3. **Leak-free pipeline** for every model, fitted on training data only:
   `SimpleImputer(median) → StandardScaler → SMOTE → Classifier`
4. **Class imbalance:** SMOTE is applied to the training data only, inside an `imblearn` pipeline, so it is also confined to the training folds during cross-validation. `class_weight="balanced"` was compared as an alternative.
5. **Evaluation:** 5-fold stratified cross-validation on the training set, plus a final evaluation on the untouched test set.
6. **Model selection:** a recall-aware rule. A candidate needs recall ≥ 0.70 in both CV and test; among those, the model with the highest mean of the five metrics averaged over CV and test is chosen.
7. **Interpretation:** impurity-based feature importance, logistic-regression coefficients and SHAP values.

## KDD Workflow

```
Problem Definition → Data Selection → Data Understanding → Data Cleaning → Data Transformation
→ Exploratory Data Analysis → Model Construction → Model Evaluation → Interpretation → Knowledge / Insights
```

![KDD workflow](results/figures/01_kdd_workflow.png)

## Models Used

| Model | Key settings |
|---|---|
| Logistic Regression | `max_iter=1000` |
| K-Nearest Neighbors | `n_neighbors=15` |
| Decision Tree | `max_depth=5`, `min_samples_leaf=5` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=2` |
| XGBoost | `n_estimators=200`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` |

## Evaluation Metrics

Accuracy, Precision, Recall (sensitivity), F1 score and ROC-AUC, plus specificity, confusion matrices and ROC curves. Recall gets special attention because a missed diabetic patient (false negative) is the costliest error in a screening context.

## Results

All numbers below come from executing `notebooks/Diabetes_Onset_Prediction.ipynb`. They are saved in `results/model_comparison.csv` and `results/metrics.csv`.

### Test-set comparison (154 patients, sorted by ROC-AUC)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.7532 | 0.6250 | 0.7407 | 0.6780 | 0.8263 |
| Logistic Regression | 0.7078 | 0.5738 | 0.6481 | 0.6087 | 0.8113 |
| **Random Forest** | **0.7468** | **0.6154** | **0.7407** | **0.6723** | **0.8094** |
| K-Nearest Neighbors | 0.6948 | 0.5455 | 0.7778 | 0.6412 | 0.7974 |
| Decision Tree | 0.7273 | 0.5833 | 0.7778 | 0.6667 | 0.7616 |

### 5-fold cross-validation on the training set (mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.772 ± 0.015 | 0.658 ± 0.026 | 0.724 ± 0.045 | 0.689 ± 0.021 | 0.843 ± 0.020 |
| K-Nearest Neighbors | 0.728 ± 0.054 | 0.588 ± 0.070 | 0.790 ± 0.020 | 0.672 ± 0.047 | 0.820 ± 0.045 |
| Decision Tree | 0.697 ± 0.029 | 0.560 ± 0.047 | 0.687 ± 0.069 | 0.612 ± 0.013 | 0.762 ± 0.013 |
| Random Forest | 0.779 ± 0.009 | 0.672 ± 0.022 | 0.715 ± 0.028 | 0.692 ± 0.009 | 0.829 ± 0.025 |
| XGBoost | 0.751 ± 0.027 | 0.632 ± 0.041 | 0.687 ± 0.058 | 0.657 ± 0.040 | 0.822 ± 0.030 |

### Final model: Random Forest

* Test set: **Accuracy 0.747, Precision 0.615, Recall 0.741, F1 0.672, ROC-AUC 0.809, Specificity 0.750**
* Confusion matrix (test): TP = 40, TN = 75, FP = 25, FN = 14
* Cross-validated ROC-AUC: 0.829 ± 0.025

**Why Random Forest and not XGBoost?** XGBoost had the best *test* ROC-AUC, accuracy, precision and F1. However, its cross-validated recall was only 0.687, below the 0.70 screening requirement. Random Forest was the only model to rank first or second on most metrics in **both** cross-validation and the test set, and it had the highest overall score under the selection rule. The margins between the top models are small; see *Limitations*.

### Most important features

Glucose is by far the most influential feature in every model (Random Forest importance 0.276; mean |SHAP| 0.146), followed by **BMI**, **Age** and **DiabetesPedigreeFunction**. Higher values of these features push predictions toward diabetes. These are predictive associations, not causal effects.

| | |
|---|---|
| ![ROC curves](results/figures/14_roc_curves.png) | ![SHAP summary](results/figures/16_shap_summary.png) |

## Project Structure

```
diabetes-onset-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── pima-indians-diabetes.csv           # dataset with header row
├── notebooks/
│   └── Diabetes_Onset_Prediction.ipynb     # main, fully executed notebook
├── src/
│   ├── preprocessing.py                    # loading, invalid-zero handling, split, imputer/scaler
│   ├── models.py                           # model definitions, SMOTE pipelines, CV helper
│   └── evaluation.py                       # metrics, confusion matrices, ROC curves
├── reports/
│   └── Diabetes_Onset_Prediction_Report.pdf  # academic project report (PDF)
├── presentation/
│   └── Diabetes_Onset_Prediction_Presentation.md  # 12-slide presentation with speaker notes
└── results/
    ├── figures/                            # 17 figures generated by the notebook
    ├── model_comparison.csv                # main metric table (test set)
    └── metrics.csv                         # detailed metrics incl. confusion counts and CV scores
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <your-repository-url>
cd diabetes-onset-prediction
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Notebook

Interactive:

```bash
jupyter notebook notebooks/Diabetes_Onset_Prediction.ipynb
```

Then choose **Kernel → Restart & Run All**.

Non-interactive, which re-executes every cell and regenerates all figures and CSV files:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Diabetes_Onset_Prediction.ipynb
```

The full run takes well under a minute on a laptop. The results above were produced with Python 3.13, pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, imbalanced-learn 0.14.2, XGBoost 3.4.1 and SHAP 0.53.0. Other library versions may produce slightly different numbers.

## Technologies Used

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · imbalanced-learn · XGBoost · SHAP · Jupyter Notebook

## Limitations

* Small dataset (768 patients; 154 in the test set), so metrics carry considerable uncertainty.
* Narrow demographics (Pima Indian women aged 21 or older); results may not generalise to other populations.
* Historical data with a possible sampling bias from one population study.
* Heavy missingness in Insulin (48.7 %) and SkinThickness (29.6 %), filled by median imputation.
* SMOTE creates synthetic, not real, patients.
* No extensive hyperparameter tuning, a fixed 0.5 threshold, uncalibrated probabilities and no external validation.
* A predictive model is not a medical diagnosis, and feature importance is not causation.

## Future Scope

Larger and more diverse datasets, external validation, cross-validated hyperparameter optimisation, per-patient explanations (SHAP waterfall plots), probability calibration and threshold selection, additional clinical features (e.g. HbA1c), deployment as a screening-support tool, and monitoring for model drift.

## Disclaimer

This repository is an academic project created for a Predictive Analytics course. It is intended for learning and research only. It is **not** a medical device or diagnostic system, it has not been clinically validated, and it must not be used to diagnose, treat or make decisions about any person's health. Consult a qualified healthcare professional for medical advice.

## License

Code is released under the MIT License (see `LICENSE`). The Pima Indians Diabetes Dataset is subject to the terms of its original providers.
