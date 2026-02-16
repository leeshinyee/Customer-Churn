import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

# 1. Load your dataset
train_df = pd.read_csv("C:\\Users\\Shin Yee\\Downloads\\Customer Churn\\customer_churn_dataset-training-master.csv")

# Drop rows with missing Churn values
train_df = train_df.dropna(subset=["Churn"])

train_df = train_df.sample(frac=0.75, random_state=42).reset_index(drop=True)

# 2. Separate features & target
X = train_df.drop(["CustomerID", "Churn"], axis=1)
y = train_df["Churn"]

# 2. Separate features & target
X = train_df.drop(["CustomerID", "Churn"], axis=1)
y = train_df["Churn"]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)
# 3. Split into train/test (hold-out validation set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Define Random Forest
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=100,
    max_depth=10
)

# 5. Cross-validation on training set
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1")
print("CV F1 scores:", cv_scores)
print("Mean CV F1:", cv_scores.mean())

# 6. Hyperparameter tuning (RandomizedSearchCV with CV)
param_dist = {
    'n_estimators': [300, 350, 400],
    'max_depth': [3, 5, 10],
    'min_samples_leaf': [10, 20, 50],
    'min_samples_split': [2],
    'max_features': ['log2', None]
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=5,                # 可以增加迭代次数以更全面搜索
    scoring="f1",
    cv=cv,
    n_jobs= 1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)

# 7. Retrain best model on full training split
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

# 8. Final Evaluation on hold-out test set
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nTest ROC-AUC:", roc_auc_score(y_test, y_proba))

# 9. F1 Scores
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_binary = f1_score(y_test, y_pred, average='binary')

print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"F1 Score (Binary - churn=1): {f1_binary:.4f}")






