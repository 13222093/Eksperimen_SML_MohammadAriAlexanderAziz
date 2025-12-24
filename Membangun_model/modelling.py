"""
Basic ML Modelling with MLflow Autolog
Author: Mohammad Ari Alexander Aziz
Description: Train machine learning models using MLflow autolog (Kriteria 2 - Basic)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(file_path='ethereum_fraud_preprocessing.csv'):
    """Load preprocessed data"""
    print(f"Loading preprocessed data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def prepare_data(df, test_size=0.2, apply_smote=True):
    """Prepare data for modeling"""
    # Separate features and target
    X = df.drop('FLAG', axis=1)
    y = df['FLAG']

    print(f"\nFeatures: {X.shape[1]}")
    print(f"Samples: {len(y)}")
    print(f"Fraud cases: {y.sum()} ({y.mean()*100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler = PowerTransformer()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE if requested
    if apply_smote:
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {X_train_scaled.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_random_forest_autolog(X_train, X_test, y_train, y_test):
    """Train Random Forest with MLflow autolog"""
    print("\n" + "="*70)
    print("Training Random Forest with MLflow Autolog")
    print("="*70)

    # Set experiment name
    mlflow.set_experiment("Ethereum_Fraud_Detection_Basic")

    # Enable autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        # Predictions
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\nRandom Forest Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        # MLflow will automatically log parameters, metrics, and model
        print("\nMLflow autolog has recorded all parameters, metrics, and model artifacts.")

    return rf_model


def train_xgboost_autolog(X_train, X_test, y_train, y_test):
    """Train XGBoost with MLflow autolog"""
    print("\n" + "="*70)
    print("Training XGBoost with MLflow Autolog")
    print("="*70)

    # Set experiment name
    mlflow.set_experiment("Ethereum_Fraud_Detection_Basic")

    # Enable autolog for XGBoost
    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name="XGBoost_Autolog"):
        # Train model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

        xgb_model.fit(X_train, y_train)

        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_proba = xgb_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\nXGBoost Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        print("\nMLflow autolog has recorded all parameters, metrics, and model artifacts.")

    return xgb_model


def main():
    """Main function"""
    print("="*70)
    print("Ethereum Fraud Detection - MLflow Autolog Training")
    print("="*70)

    # Load data
    df = load_preprocessed_data()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, apply_smote=True)

    # Train Random Forest
    rf_model = train_random_forest_autolog(X_train, X_test, y_train, y_test)

    # Train XGBoost
    xgb_model = train_xgboost_autolog(X_train, X_test, y_train, y_test)

    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print("\nTo view results, run:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")
    print("\nMake sure to take screenshots of:")
    print("  1. MLflow Dashboard (screenshoot_dashboard.jpg)")
    print("  2. Model Artifacts (screenshoot_artifak.jpg)")
    print("="*70)


if __name__ == "__main__":
    main()
