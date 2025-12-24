# Membangun Model - Kriteria 2

## Overview
This folder contains the model training scripts for Ethereum Fraud Detection using MLflow.

## Files
- `modelling.py` - Basic training with MLflow autolog (Kriteria 2 - Basic)
- `modelling_tuning.py` - Advanced training with manual logging and hyperparameter tuning (Kriteria 2 - Skilled)
- `ethereum_fraud_preprocessing.csv` - Preprocessed dataset
- `scaler.pkl` - Fitted PowerTransformer scaler
- `requirements.txt` - Python dependencies

## How to Run

### Prerequisites
Make sure you have the virtual environment activated:
```bash
# From Eksperimen_SML_MohammadAri folder
source venv/Scripts/activate  # On Windows with Git Bash
# OR
venv\Scripts\activate.bat  # On Windows CMD
# OR
venv\Scripts\Activate.ps1  # On Windows PowerShell
```

### Option 1: Run Basic Training (MLflow Autolog)
```bash
cd Membangun_model
python modelling.py
```

This will:
- Train Random Forest and XGBoost models
- Use MLflow autolog to automatically track parameters and metrics
- Save results to MLflow tracking server

### Option 2: Run Advanced Training with Tuning (RECOMMENDED for Skilled level)
```bash
cd Membangun_model
python modelling_tuning.py
```

This will:
- Perform hyperparameter tuning using GridSearchCV
- Use manual logging to track all metrics (same as autolog + additional artifacts)
- Save confusion matrices, feature importance plots, and classification reports
- Save best models as pickle files
- Log everything to MLflow

**Note:** This script may take 5-15 minutes to complete due to grid search.

### View Results in MLflow UI
After running either script, start the MLflow UI:
```bash
mlflow ui
```

Then open your browser and go to: `http://localhost:5000`

## Required Screenshots

You need to take 2 screenshots for submission:

### 1. `screenshoot_dashboard.jpg`
- Open MLflow UI (http://localhost:5000)
- Navigate to the experiment "Ethereum_Fraud_Detection_Tuning"
- Take a screenshot showing:
  - List of all runs
  - Run names
  - Metrics (accuracy, precision, recall, f1_score, roc_auc)
  - Parameters

**Tips:**
- Make sure your username/name is visible in the browser or window title
- Show multiple runs to demonstrate you've tested both models

### 2. `screenshoot_artifak.jpg`
- Click on one of the runs (e.g., "XGBoost_Tuned")
- Go to the "Artifacts" tab
- Take a screenshot showing:
  - Model artifacts
  - Confusion matrix image
  - Feature importance plot
  - Classification report
  - Saved model files

**Tips:**
- You can click on images to preview them
- Make sure the artifacts section is expanded and visible

## Expected Outputs

After running `modelling_tuning.py`, you should have:

### In MLflow:
- 2 experiment runs (RandomForest_Tuned, XGBoost_Tuned)
- Logged parameters (hyperparameters, dataset info)
- Logged metrics (accuracy, precision, recall, f1, ROC-AUC, best_cv_score)
- Logged artifacts (confusion matrices, feature importance, models)

### In the folder:
- `best_random_forest.pkl` - Best Random Forest model
- `best_xgboost.pkl` - Best XGBoost model
- `scaler_model.pkl` - Scaler for new predictions
- `confusion_matrix_*.png` - Confusion matrix plots
- `feature_importance_*.png` - Feature importance plots
- `classification_report_*.txt` - Detailed classification reports

## For Submission

Ensure your `Membangun_model` folder contains:
- ✓ `modelling.py`
- ✓ `modelling_tuning.py`
- ✓ `ethereum_fraud_preprocessing.csv`
- ✓ `requirements.txt`
- ✓ `screenshoot_dashboard.jpg` (YOU NEED TO CREATE THIS)
- ✓ `screenshoot_artifak.jpg` (YOU NEED TO CREATE THIS)
- ✓ Optionally: model pickle files and plots

## Notes
- For **Basic (2 pts)**: Run `modelling.py` (uses autolog, no tuning)
- For **Skilled (3 pts)**: Run `modelling_tuning.py` (manual logging, with tuning) ← **RECOMMENDED**
- The skilled version provides more detailed tracking and better models
