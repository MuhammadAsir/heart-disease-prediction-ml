# 🫀 Heart Disease Prediction using Machine Learning

A machine learning project for classifying whether a patient has **heart disease** based on clinical features from the **UCI Heart Disease dataset**.

The project focuses not only on model performance, but also on **clean preprocessing, preventing data leakage, baseline comparison, hyperparameter tuning, model evaluation, and feature importance analysis**.

---

## 📌 Project Overview

The goal of this project is to build a binary classification model that predicts:

* `0` → No Disease
* `1` → Disease

The original `num` target contains disease severity levels from 0–4. For this project, all positive disease levels (`1–4`) are converted into a single positive class.

Two models were evaluated:

* **Random Forest Classifier** — final tuned model
* **Logistic Regression** — baseline model

Random Forest was selected because it achieved a higher ROC AUC on the held-out test set than the Logistic Regression baseline.

> **Note:** This is an educational machine learning project and is not intended for clinical diagnosis or medical decision-making.

---

## 📊 Dataset

The project uses the **UCI Heart Disease dataset**.

The dataset contains clinical information such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Number of major vessels
* Thalassemia
* Resting ECG
* Fasting blood sugar
* ST slope

The dataset contains **920 patient records and 13 clinical features** used for modeling.

---

## 🔄 Machine Learning Pipeline

The project follows this workflow:

```text
Raw Dataset
     ↓
Remove Duplicates
     ↓
Convert Target to Binary
     ↓
Remove ID / Dataset Columns
     ↓
Train-Test Split (80/20)
     ↓
Preprocessing Pipeline
     ├── Numerical Features
     │      ├── Median Imputation
     │      └── Standard Scaling
     │
     └── Categorical Features
            ├── Most-Frequent Imputation
            └── One-Hot Encoding
     ↓
Random Forest Hyperparameter Search
     ↓
Model Evaluation
     ↓
Baseline Comparison
     ↓
Permutation Importance
     ↓
Save Trained Model
```

---

## 🧹 Data Preprocessing

A `ColumnTransformer` is used to apply different preprocessing steps to numerical and categorical features.

### Numerical Features

```python
numerical_col = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]
```

Processing:

1. Missing values → median imputation
2. Features → standard scaling

### Categorical Features

```python
nominal_cat = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]
```

Processing:

1. Missing values → most-frequent imputation
2. Categorical variables → one-hot encoding

The preprocessing is included inside the Scikit-learn pipeline so that transformations are learned only from the training data during model fitting and cross-validation.

---

## 🌲 Random Forest Model

The final model uses:

```python
RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)
```

Hyperparameters were optimized using `RandomizedSearchCV`.

### Hyperparameters searched

* `n_estimators`
* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

The search tested **40 randomly selected combinations** using **5-fold Stratified Cross-Validation**.

The tuning objective was:

```text
Accuracy
```

---

## 📈 Model Performance

### Random Forest

| Metric            |      Score |
| ----------------- | ---------: |
| Training Accuracy |     91.30% |
| Test Accuracy     | **87.50%** |
| Precision         | **86.24%** |
| Recall            | **92.16%** |
| F1 Score          | **89.10%** |
| ROC AUC           | **92.74%** |

The model correctly identified **94 of 102 disease cases** in the test set.

### Confusion Matrix

```text
[[67 15]
 [ 8 94]]
```

|                       | Predicted No Disease | Predicted Disease |
| --------------------- | -------------------: | ----------------: |
| **Actual No Disease** |                   67 |                15 |
| **Actual Disease**    |                    8 |                94 |

This means:

* **67** → True Negatives
* **15** → False Positives
* **8** → False Negatives
* **94** → True Positives

The Disease class achieved a recall of approximately **92.16%**, meaning the model identified 94 out of 102 disease cases in the test set.

---

## ⚖️ Random Forest vs Logistic Regression

Logistic Regression was used as a simpler baseline to determine whether the additional complexity of Random Forest provided a meaningful performance improvement.

### Logistic Regression

| Metric        |           Score |
| ------------- | --------------: |
| CV ROC AUC    | 0.8850 ± 0.0216 |
| Test Accuracy |          83.15% |
| Test ROC AUC  |          0.9041 |
| Test F1       |          0.8502 |

### Comparison

| Model               | Test Accuracy | Test ROC AUC |    Test F1 |
| ------------------- | ------------: | -----------: | ---------: |
| Logistic Regression |        83.15% |       0.9041 |     0.8502 |
| **Random Forest**   |    **87.50%** |   **0.9274** | **0.8910** |

Random Forest achieved a higher test ROC AUC than the Logistic Regression baseline:

```text
Random Forest       → 0.9274
Logistic Regression → 0.9041
```

This provides evidence that the additional model complexity was useful for this dataset.

---

## 🔍 Feature Importance

Two approaches were used to investigate which features influenced the Random Forest model.

### 1. Impurity-Based Feature Importance

Random Forest provides feature importance based on impurity reduction.

However, this type of importance can be biased toward certain types of features, particularly continuous or high-cardinality features.

Therefore, it was not used as the only interpretability method.

### 2. Permutation Importance

Permutation importance was calculated on the **held-out test set** using ROC AUC as the scoring metric.

The idea is simple:

> Shuffle one feature and measure how much the model's performance decreases.

A larger decrease means that the model relied more heavily on that feature for its predictions.

### Top Features

| Feature   | Mean Importance |    Std |
| --------- | --------------: | -----: |
| `cp`      |          0.0673 | 0.0172 |
| `chol`    |          0.0376 | 0.0102 |
| `exang`   |          0.0263 | 0.0072 |
| `oldpeak` |          0.0244 | 0.0072 |
| `sex`     |          0.0157 | 0.0062 |
| `thalch`  |          0.0083 | 0.0064 |
| `thal`    |          0.0058 | 0.0047 |

The `cp` feature, representing **chest pain type**, had the largest permutation importance in this experiment.

> Feature importance indicates how the trained model used the available features. It does **not** establish that a feature causes heart disease or that it is independently medically predictive.

---

## 📊 Visualizations

The project generates the following visualizations:

### Confusion Matrix

`best_confusion_matrix.png`

Shows the distribution of correct and incorrect predictions.

### ROC Curve

`best_roc_curve.png`

Shows the model's ability to distinguish between the two classes across different classification thresholds.

The resulting ROC AUC is:

```text
0.9274
```

### Impurity-Based Feature Importance

`best_feature_importance.png`

Shows the Random Forest's built-in feature importance values.

### Permutation Importance

`best_permutation_importance.png`

Shows the decrease in test ROC AUC when individual input features are randomly shuffled.

---

## 🛡️ Leakage Prevention

Data leakage was considered during the pipeline design.

The following columns were removed:

```python
df.drop(columns=["id", "dataset"], inplace=True)
```

`id` was removed because it is simply an identifier.

`dataset` was removed because it represents the hospital/source of the observation and could introduce dataset-specific information rather than meaningful patient-level information.

Additionally, imputation, scaling, and one-hot encoding are performed inside the Scikit-learn pipeline. This ensures preprocessing parameters are learned from the appropriate training folds rather than from the entire dataset before cross-validation.

---

## 🧪 Evaluation Strategy

The dataset was divided using:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produced:

```text
80% → Training
20% → Testing
```

The training set was used for:

* Preprocessing
* Hyperparameter tuning
* Cross-validation
* Model fitting

The test set was reserved for final evaluation.

---

## 💾 Saved Model

The trained pipeline is saved using `joblib`:

```text
best_heart_disease_model.joblib
```

Because the preprocessing steps are included in the pipeline, the saved model contains both the preprocessing and Random Forest model.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

### Machine Learning Techniques

* Random Forest
* Logistic Regression
* Stratified K-Fold Cross-Validation
* Randomized Hyperparameter Search
* One-Hot Encoding
* Missing Value Imputation
* Feature Scaling
* Permutation Importance
* ROC AUC
* Confusion Matrix
* Classification Metrics

---

## 📁 Project Structure

```text
heart-disease-ml/
│
├── heart_disease_uci.csv
├── heart_disease_model.py
│
├── best_heart_disease_model.joblib
│
├── best_confusion_matrix.png
├── best_roc_curve.png
├── best_feature_importance.png
├── best_permutation_importance.png
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd heart-disease-ml
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Run the project

```bash
python heart_disease_model.py
```

The script will:

1. Load and clean the dataset
2. Split the data
3. Build the preprocessing pipeline
4. Tune the Random Forest
5. Train the Logistic Regression baseline
6. Evaluate both models
7. Generate visualizations
8. Calculate permutation importance
9. Save the trained model

---

## 📌 Key Takeaways

This project helped me move beyond simply training a machine learning model.

The main lessons were:

* A model should be compared against a **simple baseline**.
* Hyperparameter tuning should be performed using **cross-validation**.
* Preprocessing should be included inside the pipeline to reduce the risk of **data leakage**.
* Accuracy alone does not provide a complete picture of classification performance.
* Recall is particularly informative when examining the model's ability to identify positive cases.
* Feature importance should be interpreted carefully rather than treated as causal evidence.
* Using **permutation importance** provides an additional perspective beyond Random Forest's built-in importance.

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.

The model has not undergone clinical validation, external validation, prospective testing, or regulatory evaluation. Its predictions should **not** be used for medical diagnosis or treatment decisions.

---

## 👨‍💻 Author

**Muhammad Asir Hossain Chowdhury**

Machine Learning / AI Engineering

If you found this project useful, feel free to ⭐ the repository.
