
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

plt.switch_backend("Agg")
RANDOM_STATE = 42

# 1. LOAD & CLEAN

df = pd.read_csv("heart_disease_uci.csv")

df.drop_duplicates(inplace=True)

# Binary target: 0 = no disease, 1 = disease (any severity 1-4 collapsed to 1)
df["num"] = (df["num"] > 0).astype(int)

# Drop 'id' (just a row number) and 'dataset' (hospital-of-origin leakage - see note above)
df.drop(columns=["id", "dataset"], inplace=True)

numerical_col = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
nominal_cat = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

x = df.drop(columns=["num"])
y = df["num"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


# 2. PREPROCESSING PIPELINE

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

nominal_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_transformer, numerical_col),
    ("nominal", nominal_transformer, nominal_cat),
])


# 3. HYPERPARAMETER SEARCH (Random Forest)
# RandomizedSearchCV samples a fixed number of random combinations instead of
# testing every single one (GridSearchCV) -- much faster while still finding
# a near-optimal setting, especially with a search space this size.
rf_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
])

param_distributions = {
    "model__n_estimators": [200, 300, 400, 500],
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 6, 8, 10, 12, 15],
    "model__min_samples_split": [2, 4, 6],
    "model__min_samples_leaf": [1, 2, 3],
    "model__max_features": ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=rf_pipe,
    param_distributions=param_distributions,
    n_iter=40,               # try 40 random hyperparameter combinations
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

search.fit(x_train, y_train)
best_model = search.best_estimator_

print("Best hyperparameters found:")
print(search.best_params_)
print(f"Best cross-validation accuracy: {search.best_score_:.4f}")



# 4. EVALUATE ON TRAIN AND TEST SETS

train_pred = best_model.predict(x_train)
test_pred = best_model.predict(x_test)
test_proba = best_model.predict_proba(x_test)[:, 1]

print("\n--- Performance ---")
print(f"Training Accuracy : {accuracy_score(y_train, train_pred):.4f}")
print(f"Testing Accuracy  : {accuracy_score(y_test, test_pred):.4f}")
print(f"Precision         : {precision_score(y_test, test_pred):.4f}")
print(f"Recall            : {recall_score(y_test, test_pred):.4f}")
print(f"F1 Score          : {f1_score(y_test, test_pred):.4f}")
print(f"ROC AUC           : {roc_auc_score(y_test, test_proba):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report:")
print(classification_report(y_test, test_pred, target_names=["No Disease", "Disease"]))



# 5. PLOTS

cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned Random Forest")
plt.savefig("best_confusion_matrix.png", bbox_inches="tight")
plt.close()

fpr, tpr, _ = roc_curve(y_test, test_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, test_proba):.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Random Forest")
plt.legend()
plt.savefig("best_roc_curve.png", bbox_inches="tight")
plt.close()

# Feature importance: which inputs mattered most to the model
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(7, 6))
sns.barplot(x="importance", y="feature", data=importance_df, color="steelblue")
plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.savefig("best_feature_importance.png", bbox_inches="tight")
plt.close()

print("\nSaved: best_confusion_matrix.png, best_roc_curve.png, best_feature_importance.png")



# 6. SAVE MODEL

joblib.dump(best_model, "best_heart_disease_model.joblib")
print("Model saved to best_heart_disease_model.joblib")
