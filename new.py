

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG — CHANGE THE PATH BELOW TO YOUR DOWNLOADED CSV
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\penum\Downloads\archive (5)\heart.csv"   # <-- Replace with your actual path
MODEL_OUTPUT  = "heart_model.pkl"
SCALER_OUTPUT = "heart_scaler.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & INSPECT
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  NeuroBee Heart Disease AI — Model Training")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
# Encode categorical columns if present (for fedesoriano dataset)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    print(f"  Encoded: {col}")

# Target column — adjust if your CSV uses a different name
TARGET = "HeartDisease"
if TARGET not in df.columns:
    # Fallback: last column
    TARGET = df.columns[-1]
    print(f"  ⚠️  'HeartDisease' not found — using last column: {TARGET}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\n  Features : {list(X.columns)}")
print(f"  Target   : {TARGET}  |  Classes: {y.unique()}")
print(f"  Positive rate: {y.mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE MODEL (Voting: GBM + RF + LR)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔧 Training ensemble model …")

gbm = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, subsample=0.8, random_state=42
)
rf  = RandomForestClassifier(
    n_estimators=300, max_depth=None,
    min_samples_split=2, random_state=42, n_jobs=-1
)
lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

ensemble = VotingClassifier(
    estimators=[('gbm', gbm), ('rf', rf), ('lr', lr)],
    voting='soft'
)
ensemble.fit(X_train_s, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
y_pred      = ensemble.predict(X_test_s)
y_prob      = ensemble.predict_proba(X_test_s)[:, 1]

acc   = accuracy_score(y_test, y_pred)
auc   = roc_auc_score(y_test, y_prob)

print("\n" + "─" * 60)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ROC-AUC   : {auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("─" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all_s = scaler.transform(X)
cv_scores = cross_val_score(ensemble, X_all_s, y, cv=cv, scoring='accuracy')
print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Folds: {[f'{s:.4f}' for s in cv_scores]}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE MODEL & SCALER
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(ensemble, MODEL_OUTPUT)
joblib.dump(scaler,   SCALER_OUTPUT)
joblib.dump(list(X.columns), "feature_names.pkl")

print(f"\n✅ Model  saved → {MODEL_OUTPUT}")
print(f"✅ Scaler saved → {SCALER_OUTPUT}")
print(f"✅ Features saved → feature_names.pkl")
print("\n🎉 Training complete! Run app.html next.\n")
