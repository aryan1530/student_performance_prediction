"""
Student Performance Prediction System
Covers: Data Preprocessing, EDA, Linear & Logistic Regression, Feature Scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)

# ─── Paths ────────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Load dataset using Pandas
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path="data/student_data.csv"):
    print("\n" + "="*60)
    print("STEP 1: Loading Dataset")
    print("="*60)
    df = pd.read_csv(path)
    print(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types:\n", df.dtypes)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – Clean and preprocess data
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_data(df):
    print("\n" + "="*60)
    print("STEP 2: Data Preprocessing")
    print("="*60)

    print(f"\nMissing values before:\n{df.isnull().sum()}")

    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded '{col}'")

    print(f"\nMissing values after:\n{df.isnull().sum()}")
    print(f"\n✅ Preprocessing complete. Shape: {df.shape}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Perform EDA using visualization
# ══════════════════════════════════════════════════════════════════════════════
def perform_eda(df):
    print("\n" + "="*60)
    print("STEP 3: Exploratory Data Analysis (EDA)")
    print("="*60)

    print("\nStatistical Summary:")
    print(df.describe().round(2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("EDA – Student Performance Dataset", fontsize=16, fontweight='bold')

    # 1. Marks distribution
    axes[0, 0].hist(df['marks'], bins=20, color='steelblue', edgecolor='white')
    axes[0, 0].set_title('Distribution of Marks')
    axes[0, 0].set_xlabel('Marks')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Pass/Fail count
    labels = ['Fail (0)', 'Pass (1)']
    values = df['pass_fail'].value_counts().sort_index()
    axes[0, 1].bar(labels, values, color=['tomato', 'mediumseagreen'], edgecolor='white')
    axes[0, 1].set_title('Pass / Fail Count')
    axes[0, 1].set_ylabel('Count')

    # 3. Study hours vs Marks
    axes[0, 2].scatter(df['study_hours_per_day'], df['marks'],
                       alpha=0.4, color='darkorange', edgecolors='none')
    axes[0, 2].set_title('Study Hours vs Marks')
    axes[0, 2].set_xlabel('Study Hours / Day')
    axes[0, 2].set_ylabel('Marks')

    # 4. Attendance vs Marks
    axes[1, 0].scatter(df['attendance_percentage'], df['marks'],
                       alpha=0.4, color='mediumpurple', edgecolors='none')
    axes[1, 0].set_title('Attendance % vs Marks')
    axes[1, 0].set_xlabel('Attendance (%)')
    axes[1, 0].set_ylabel('Marks')

    # 5. Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, ax=axes[1, 1], annot=True, fmt=".2f",
                cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
    axes[1, 1].set_title('Correlation Heatmap')

    # 6. Boxplot: marks by pass/fail
    df.boxplot(column='marks', by='pass_fail', ax=axes[1, 2],
               patch_artist=True)
    axes[1, 2].set_title('Marks by Pass/Fail')
    axes[1, 2].set_xlabel('Pass/Fail')
    axes[1, 2].set_ylabel('Marks')
    plt.sca(axes[1, 2])
    plt.xticks([1, 2], ['Fail', 'Pass'])

    plt.tight_layout()
    plt.savefig("outputs/eda_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ EDA plots saved to outputs/eda_plots.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Split data into training and testing sets
# ══════════════════════════════════════════════════════════════════════════════
def split_data(df):
    print("\n" + "="*60)
    print("STEP 4: Train / Test Split")
    print("="*60)

    feature_cols = [c for c in df.columns if c not in ['marks', 'pass_fail']]
    X = df[feature_cols]
    y_reg  = df['marks']        # for Linear Regression
    y_clf  = df['pass_fail']    # for Logistic Regression

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )

    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing  samples : {X_test.shape[0]}")
    print(f"  Features used    : {list(feature_cols)}")
    return X_train, X_test, yr_train, yr_test, yc_train, yc_test, feature_cols

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Feature Scaling + Train & Evaluate Models
# ══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(X_train, X_test, yr_train, yr_test, yc_train, yc_test):
    print("\n" + "="*60)
    print("STEP 5: Feature Scaling + Training Models")
    print("="*60)

    # ── Feature Scaling (Key Concept 4) ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("✅ Feature scaling applied (StandardScaler)")

    # ── Linear Regression (Key Concept 3a) ───────────────────────────────────
    print("\n--- Linear Regression (Marks Prediction) ---")
    lr = LinearRegression()
    lr.fit(X_train_sc, yr_train)
    yr_pred = lr.predict(X_test_sc)

    mae  = mean_absolute_error(yr_test, yr_pred)
    mse  = mean_squared_error(yr_test, yr_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(yr_test, yr_pred)

    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R²   : {r2:.4f}")

    # ── Logistic Regression (Key Concept 3b) ─────────────────────────────────
    print("\n--- Logistic Regression (Pass/Fail Classification) ---")
    log = LogisticRegression(max_iter=1000, random_state=42)
    log.fit(X_train_sc, yc_train)
    yc_pred = log.predict(X_test_sc)

    acc = accuracy_score(yc_test, yc_pred)
    print(f"  Accuracy : {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(yc_test, yc_pred, target_names=['Fail', 'Pass']))

    # ── Visualise results ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight='bold')

    # Actual vs Predicted (regression)
    axes[0].scatter(yr_test, yr_pred, alpha=0.5, color='steelblue', edgecolors='none')
    mn, mx = yr_test.min(), yr_test.max()
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect fit')
    axes[0].set_title(f'Linear Regression\nR² = {r2:.3f}  RMSE = {rmse:.2f}')
    axes[0].set_xlabel('Actual Marks')
    axes[0].set_ylabel('Predicted Marks')
    axes[0].legend()

    # Confusion Matrix (classification)
    cm = confusion_matrix(yc_test, yc_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    axes[1].set_title(f'Logistic Regression\nAccuracy = {acc*100:.1f}%')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig("outputs/model_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Model plots saved to outputs/model_results.png")

    # Save models
    pickle.dump(lr,     open("models/linear_regression.pkl",  "wb"))
    pickle.dump(log,    open("models/logistic_regression.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl",              "wb"))
    print("✅ Models saved to models/")

    return lr, log, scaler, {
        'linear': {'mae': mae, 'rmse': rmse, 'r2': r2},
        'logistic': {'accuracy': acc}
    }

# ══════════════════════════════════════════════════════════════════════════════
# Predict function (used by app.py)
# ══════════════════════════════════════════════════════════════════════════════
def predict_student(input_dict):
    lr     = pickle.load(open("models/linear_regression.pkl",  "rb"))
    log    = pickle.load(open("models/logistic_regression.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl",              "rb"))

    features = [
        'study_hours_per_day', 'attendance_percentage', 'previous_score',
        'sleep_hours', 'extra_activities', 'internet_access',
        'family_support', 'gender'
    ]
    X = pd.DataFrame([input_dict])[features]
    Xs = scaler.transform(X)

    marks     = round(float(lr.predict(Xs)[0]),  2)
    pass_fail = int(log.predict(Xs)[0])
    prob      = float(log.predict_proba(Xs)[0][1])

    return {
        'predicted_marks': max(0, min(100, marks)),
        'pass_fail':       'Pass' if pass_fail == 1 else 'Fail',
        'pass_probability': round(prob * 100, 1)
    }

# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    perform_eda(df)
    X_train, X_test, yr_train, yr_test, yc_train, yc_test, features = split_data(df)
    lr, log, scaler, metrics = train_and_evaluate(
        X_train, X_test, yr_train, yr_test, yc_train, yc_test
    )

    print("\n" + "="*60)
    print("OUTPUT – Predicted Result & Model Accuracy")
    print("="*60)
    sample = {
        'study_hours_per_day': 5, 'attendance_percentage': 80,
        'previous_score': 65, 'sleep_hours': 7,
        'extra_activities': 1, 'internet_access': 1,
        'family_support': 1, 'gender': 0
    }
    result = predict_student(sample)
    print(f"\nSample Student Prediction:")
    print(f"  Predicted Marks    : {result['predicted_marks']}")
    print(f"  Result             : {result['pass_fail']}")
    print(f"  Pass Probability   : {result['pass_probability']}%")
    print(f"\nLinear Regression  → R² = {metrics['linear']['r2']:.4f}")
    print(f"Logistic Regression → Accuracy = {metrics['logistic']['accuracy']*100:.2f}%")
    print("\n✅ Pipeline complete!")