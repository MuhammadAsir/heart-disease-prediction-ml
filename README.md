
# Heart Disease Prediction using Machine Learning

A machine learning project for predicting the presence of heart disease using patient clinical data from the UCI Heart Disease dataset.

This project focuses on building a complete machine learning pipeline with data preprocessing, Random Forest classification, hyperparameter tuning, cross-validation, and model evaluation.

## Project Objective

The main objective is to predict whether a patient has heart disease based on available clinical features.

The project also focuses on understanding:

* Data preprocessing
* Missing-value handling
* Categorical feature encoding
* Feature scaling
* Random Forest classification
* Hyperparameter tuning
* Cross-validation
* Model evaluation
* Overfitting and generalization

## Dataset

The project uses:

```text
heart_disease_uci.csv
```

The target variable is:

```text
num
```

The original `num` column contains values from `0` to `4`.

For this project, it is converted into a binary classification target:

```text
0       → No Disease
1, 2, 3, 4 → Disease
```

This transformation is performed using:

```python
df["num"] = (df["num"] > 0).astype(int)
```

## Features

### Numerical Features

```text
age
trestbps
chol
thalch
oldpeak
ca
```

### Categorical Features

```text
sex
cp
fbs
restecg
exang
slope
thal
```

The following columns are removed before training:

```text
id
dataset
```

The `id` column is an identifier, while the `dataset` column indicates the source hospital. The project removes both before model training.

## Machine Learning Workflow

```text
Heart Disease Dataset
        ↓
Remove Duplicate Rows
        ↓
Convert Target to Binary
        ↓
Remove ID and Dataset Columns
        ↓
Train/Test Split
        ↓
Handle Missing Values
        ↓
Encode Categorical Features
        ↓
Scale Numerical Features
        ↓
Random Forest
        ↓
Hyperparameter Tuning
        ↓
5-Fold Cross-Validation
        ↓
Final Model
        ↓
Model Evaluation
        ↓
Save Model & Visualizations
```

## Train-Test Split

The dataset is divided into training and testing data using an 80/20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)
```

The project uses:

```python
RANDOM_STATE = 42
```

`stratify=y` is used to preserve the class distribution between the training and testing sets.

## Data Preprocessing

The project uses a `ColumnTransformer` to apply different preprocessing to numerical and categorical features.

### Numerical Features

For numerical features:

1. Missing values are replaced using the median.
2. Features are standardized using `StandardScaler`.

```python
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

### Categorical Features

For categorical features:

1. Missing values are replaced with the most frequent value.
2. Features are converted into numerical representations using `OneHotEncoder`.

```python
nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
```

These transformations are combined using `ColumnTransformer`.

## Model

The final model is a:

**Random Forest Classifier**

The model is configured with:

```python
RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced"
)
```

The `balanced` class weighting helps the model account for differences in class distribution.

## Hyperparameter Tuning

Instead of using a Random Forest with default parameters, the project uses `RandomizedSearchCV` to search for better hyperparameter combinations.

The search includes:

```text
n_estimators
criterion
max_depth
min_samples_split
min_samples_leaf
max_features
```

The project tests:

```text
40 random hyperparameter combinations
```

using:

```text
5-fold Stratified Cross-Validation
```

with accuracy as the scoring metric.

## Overfitting and Generalization

A key part of this project was paying attention to the difference between training and testing performance.

Final results:

```text
Training Accuracy : 91.30%
Testing Accuracy  : 87.50%
```

The difference is:

```text
91.30% - 87.50% = 3.80 percentage points
```

This train/test gap was useful for understanding how the model performs on data it has not seen during training.

Hyperparameter tuning was used as part of the effort to build a better-performing and better-controlled Random Forest model. The project evaluates performance on a held-out test set rather than relying only on training performance.

## Final Model Performance

```text
Training Accuracy : 91.30%
Testing Accuracy  : 87.50%
Precision         : 86.24%
Recall            : 92.16%
F1 Score          : 89.10%
ROC AUC           : 92.74%
```

### What these metrics mean

**Accuracy**
Percentage of all test predictions that were correct.

**Precision**
Among the patients predicted as having heart disease, how many were actually positive.

**Recall**
Among the patients who actually had heart disease, how many were correctly identified.

**F1 Score**
A combined measure of precision and recall.

**ROC-AUC**
Measures how well the model separates the two classes across different classification thresholds.

## Confusion Matrix

The project generates a confusion matrix to show:

* Correct No Disease predictions
* Incorrect Disease predictions
* Incorrect No Disease predictions
* Correct Disease predictions

The confusion matrix is saved as:

```text
best_confusion_matrix.png
```

## ROC Curve

The project also generates an ROC curve using the predicted probabilities from the Random Forest model.

The ROC-AUC value is:

```text
0.9274
```

The figure is saved as:

```text
best_roc_curve.png
```

## Feature Importance

Random Forest provides feature importance values that can be used to see which transformed input features contributed most to the model.

The project generates a top-15 feature importance visualization:

```text
best_feature_importance.png
```

The feature importance values are extracted from the trained Random Forest model.

## Model Saving

The final trained pipeline is saved using Joblib:

```python
joblib.dump(best_model, "best_heart_disease_model.joblib")
```

This saves the trained model together with its preprocessing pipeline so it can be loaded and reused later.

## Project Files

The simplest project structure is:

```text
heart-disease-prediction/
│
├── heart_disease_prediction.py
├── heart_disease_uci.csv
├── best_confusion_matrix.png
├── best_roc_curve.png
├── best_feature_importance.png
└── best_heart_disease_model.joblib
```

The Python file and CSV dataset can be kept in the **same folder**.

## How to Run

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Then place:

```text
heart_disease_prediction.py
heart_disease_uci.csv
```

in the same folder.

Run:

```bash
python heart_disease_prediction.py
```

The program will:

1. Load the dataset
2. Clean the data
3. Prepare the features
4. Perform preprocessing
5. Tune the Random Forest
6. Train the final model
7. Evaluate the model
8. Generate visualizations
9. Save the trained model

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**

## Key Learning Outcomes

Through this project, I practiced:

* Data cleaning
* Missing-value imputation
* Feature preprocessing
* One-hot encoding
* Feature scaling
* Machine learning pipelines
* Random Forest classification
* Hyperparameter tuning
* Stratified cross-validation
* Classification evaluation
* Overfitting and generalization
* Model persistence

## Previous Approach

I previously worked on the same dataset using **Logistic Regression**, where the model achieved **85.33% accuracy**.

Revisiting the same dataset with Random Forest gave me an opportunity to apply concepts I learned later in my Machine Learning journey, including hyperparameter tuning and cross-validation.

This comparison helped me understand that improving a machine learning project is not only about changing the algorithm. The preprocessing pipeline, validation strategy, hyperparameters, and evaluation process also matter.

## Important Note

This project is intended for **educational and portfolio purposes**.

It is **not a clinical diagnostic system** and should not be used to make medical decisions. The reported metrics are based on the specific dataset, preprocessing pipeline, train/test split, and experimental setup used in this project.
