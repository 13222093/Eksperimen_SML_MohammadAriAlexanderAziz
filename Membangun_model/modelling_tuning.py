"""
Advanced ML Modelling with MLflow Manual Logging and Hyperparameter Tuning
Author: Mohammad Ari Alexander Aziz
Description: Train ML models with hyperparameter tuning using MLflow manual logging (Kriteria 2 - Skilled)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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

    # Save scaler
    with open('scaler_model.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Apply SMOTE if requested
    if apply_smote:
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {X_train_scaled.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test


def save_confusion_matrix_plot(y_true, y_pred, model_name):
    """Save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def save_feature_importance_plot(model, feature_names, model_name):
    """Save feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices],
                   rotation=90)
        plt.tight_layout()

        filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    return None


def train_random_forest_tuned(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest with hyperparameter tuning and manual logging"""
    print("\n" + "="*70)
    print("Training Random Forest with Hyperparameter Tuning")
    print("="*70)

    # Set experiment
    mlflow.set_experiment("Ethereum_Fraud_Detection_Tuning")

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    print("\nHyperparameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    with mlflow.start_run(run_name="RandomForest_Tuned"):
        # Log dataset info
        mlflow.log_param("dataset_name", "ethereum_fraud_preprocessing.csv")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("smote_applied", "True")

        # Grid search
        print("\nPerforming Grid Search...")
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, scoring='f1',
            verbose=1, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Best model
        best_rf = grid_search.best_estimator_

        # Log best parameters (manual logging)
        print("\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
            mlflow.log_param(f"best_{param}", value)

        # Predictions
        y_pred = best_rf.predict(X_test)
        y_proba = best_rf.predict_proba(X_test)[:, 1]

        # Calculate metrics (manual logging - same as autolog)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        print(f"\nRandom Forest Tuned Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")

        # Save and log confusion matrix
        cm_file = save_confusion_matrix_plot(y_test, y_pred, "Random Forest")
        mlflow.log_artifact(cm_file)

        # Save and log feature importance
        fi_file = save_feature_importance_plot(best_rf, feature_names, "Random Forest")
        if fi_file:
            mlflow.log_artifact(fi_file)

        # Log classification report as text
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        with open('classification_report_rf.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report_rf.txt')

        # Log model
        mlflow.sklearn.log_model(best_rf, "random_forest_model")

        # Save model locally
        with open('best_random_forest.pkl', 'wb') as f:
            pickle.dump(best_rf, f)
        mlflow.log_artifact('best_random_forest.pkl')

        print("\nAll metrics, parameters, and artifacts logged to MLflow.")

    return best_rf


def train_xgboost_tuned(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost with hyperparameter tuning and manual logging"""
    print("\n" + "="*70)
    print("Training XGBoost with Hyperparameter Tuning")
    print("="*70)

    # Set experiment
    mlflow.set_experiment("Ethereum_Fraud_Detection_Tuning")

    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }

    print("\nHyperparameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    with mlflow.start_run(run_name="XGBoost_Tuned"):
        # Log dataset info
        mlflow.log_param("dataset_name", "ethereum_fraud_preprocessing.csv")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        mlflow.log_param("smote_applied", "True")

        # Grid search
        print("\nPerforming Grid Search...")
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=5, scoring='f1',
            verbose=1, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Best model
        best_xgb = grid_search.best_estimator_

        # Log best parameters
        print("\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
            mlflow.log_param(f"best_{param}", value)

        # Predictions
        y_pred = best_xgb.predict(X_test)
        y_proba = best_xgb.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        print(f"\nXGBoost Tuned Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")

        # Save and log confusion matrix
        cm_file = save_confusion_matrix_plot(y_test, y_pred, "XGBoost")
        mlflow.log_artifact(cm_file)

        # Save and log feature importance
        fi_file = save_feature_importance_plot(best_xgb, feature_names, "XGBoost")
        if fi_file:
            mlflow.log_artifact(fi_file)

        # Log classification report as text
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
        with open('classification_report_xgb.txt', 'w') as f:
            f.write(report)
        mlflow.log_artifact('classification_report_xgb.txt')

        # Log model
        mlflow.sklearn.log_model(best_xgb, "xgboost_model")

        # Save model locally
        with open('best_xgboost.pkl', 'wb') as f:
            pickle.dump(best_xgb, f)
        mlflow.log_artifact('best_xgboost.pkl')

        print("\nAll metrics, parameters, and artifacts logged to MLflow.")

    return best_xgb


def main():
    """Main function"""
    print("="*70)
    print("Ethereum Fraud Detection - MLflow Manual Logging with Tuning")
    print("="*70)

    # Load data
    df = load_preprocessed_data()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, apply_smote=True)

    # Get feature names
    feature_names = df.drop('FLAG', axis=1).columns.tolist()

    # Train Random Forest with tuning
    rf_model = train_random_forest_tuned(X_train, X_test, y_train, y_test, feature_names)

    # Train XGBoost with tuning
    xgb_model = train_xgboost_tuned(X_train, X_test, y_train, y_test, feature_names)

    print("\n" + "="*70)
    print("Training Completed!")
    print("="*70)
    print("\nModels saved:")
    print("  - best_random_forest.pkl")
    print("  - best_xgboost.pkl")
    print("  - scaler_model.pkl")
    print("\nArtifacts created:")
    print("  - Confusion matrices")
    print("  - Feature importance plots")
    print("  - Classification reports")
    print("\nTo view results, run:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")
    print("\nMake sure to take screenshots of:")
    print("  1. MLflow Dashboard showing all runs (screenshoot_dashboard.jpg)")
    print("  2. Model Artifacts page (screenshoot_artifak.jpg)")
    print("="*70)


if __name__ == "__main__":
    main()
