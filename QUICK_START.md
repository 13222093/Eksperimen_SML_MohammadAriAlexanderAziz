# Quick Start Guide - Machine Learning Submission

**Your GitHub username:** `13222093`

Follow these steps IN ORDER. Don't skip ahead!

---

## STEP 1: Push to GitHub (15 minutes)

### 1A. Push Repository #1 - Eksperimen

Open **Git Bash** and run these commands ONE BY ONE:

```bash
# Go to the folder
cd "/c/Users/Ari Azis/Downloads/GitHub_Repos_ToUpload/Eksperimen_SML_MohammadAri"

# Initialize git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Ethereum fraud detection preprocessing"

# Rename branch to main
git branch -M main

# Add your GitHub repo
git remote add origin https://github.com/13222093/Eksperimen_SML_MohammadAri.git

# Push to GitHub
git push -u origin main
```

✅ **Check:** Go to https://github.com/13222093/Eksperimen_SML_MohammadAri - you should see files!

---

### 1B. Push Repository #2 - Workflow-CI

Open **Git Bash** and run these commands ONE BY ONE:

```bash
# Go to the folder
cd "/c/Users/Ari Azis/Downloads/GitHub_Repos_ToUpload/Workflow-CI"

# Initialize git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: MLflow Project with CI/CD"

# Rename branch to main
git branch -M main

# Add your GitHub repo
git remote add origin https://github.com/13222093/Workflow-CI.git

# Push to GitHub
git push -u origin main
```

✅ **Check:** Go to https://github.com/13222093/Workflow-CI - you should see files!

---

### 1C. Update the .txt Files

**Good news:** Already updated with your username (13222093)! ✅

You can verify:
- `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Eksperimen_SML_MohammadAri.txt`
- `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Workflow-CI.txt`

---

## STEP 2: Take MLflow Screenshots (30 minutes)

### 2A. Fix the XGBoost Bug First! (IMPORTANT)

Before running training, fix this bug:

1. Open: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Membangun_model\modelling_tuning.py`

2. Go to **line 304** (use Ctrl+G in most editors)

3. Find this line:
   ```python
   mlflow.xgboost.log_model(best_xgb, "xgboost_model")
   ```

4. Change it to:
   ```python
   mlflow.sklearn.log_model(best_xgb, "xgboost_model")
   ```
   (Only change `xgboost` to `sklearn`)

5. **Save the file** (Ctrl+S)

---

### 2B. Run Training Script

Open **Git Bash**:

```bash
# Go to Membangun_model folder
cd "/c/Users/Ari Azis/Downloads/SMSML_MohammadAri/Membangun_model"

# Activate virtual environment
source "/c/Users/Ari Azis/Downloads/Submisi asah sistem ML/Eksperimen_SML_MohammadAri/venv/Scripts/activate"

# Run training (takes 10-15 minutes)
python modelling_tuning.py
```

⏰ **Wait** for training to complete. You'll see:
- "Training Random Forest with Hyperparameter Tuning"
- "Training XGBoost with Hyperparameter Tuning"
- "Training Completed!" at the end

✅ **Check:** You should see new files created:
- `best_random_forest.pkl`
- `best_xgboost.pkl`
- `scaler_model.pkl`
- `confusion_matrix_*.png`
- `feature_importance_*.png`

---

### 2C. Start MLflow UI

In the **SAME terminal** after training completes:

```bash
# Start MLflow UI
mlflow ui
```

You'll see: `Listening at: http://127.0.0.1:5000`

---

### 2D. Take Screenshots

1. Open browser: **http://localhost:5000**
2. You'll see a list of experiment runs (2 runs: RandomForest_Tuned and XGBoost_Tuned)

**Screenshot 1:**
- Take screenshot of this page (showing the 2 runs with their F1 scores, accuracy, etc.)
- Save as: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Membangun_model\screenshoot_dashboard.jpg`

**Screenshot 2:**
- Click on one of the runs (either RandomForest_Tuned or XGBoost_Tuned)
- Click the **"Artifacts"** tab
- You'll see: confusion_matrix.png, feature_importance.png, classification_report.txt, model files
- Take screenshot of this page (showing all the artifacts)
- Save as: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Membangun_model\screenshoot_artifak.jpg`

✅ **Done with Kriteria 2!**

Press `Ctrl+C` in the terminal to stop MLflow UI.

---

## STEP 3: Take Monitoring Screenshots (60 minutes)

### 3A. Install Prometheus & Grafana (one-time setup)

**Prometheus:**
1. Download: https://prometheus.io/download/ (get the Windows .zip file)
2. Extract to: `C:\Users\Ari Azis\Downloads\prometheus-3.8.1.windows-amd64`
3. You should already have this! ✅

**Grafana:**
1. Download: https://grafana.com/grafana/download?platform=windows (get Windows installer)
2. Install it (default options)
3. It runs on port 3000

---

### 3B. Fix Prometheus Configuration (IMPORTANT!)

Before starting Prometheus, fix the port number:

1. Open: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Monitoring dan Logging\2.prometheus.yml`

2. Find the line with `localhost:8000`

3. Change it to `localhost:5001`:

   **FROM:**
   ```yaml
   - targets: ['localhost:8000']
   ```

   **TO:**
   ```yaml
   - targets: ['localhost:5001']
   ```

4. **Save the file**

---

### 3C. Copy Model Files (IMPORTANT!)

The Inference API needs the trained models:

```bash
# Copy model files from Membangun_model to Monitoring dan Logging
cd "/c/Users/Ari Azis/Downloads/SMSML_MohammadAri/Monitoring dan Logging"

cp "../Membangun_model/best_xgboost.pkl" .
cp "../Membangun_model/scaler_model.pkl" .
```

✅ **Check:** You should now have these files in `Monitoring dan Logging/`:
- `best_xgboost.pkl`
- `scaler_model.pkl`

---

### 3D. Start 3 Services (You Need 3 Terminals!)

**Terminal 1 - Start Prometheus:**

```bash
# Go to Prometheus folder
cd "/c/Users/Ari Azis/Downloads/prometheus-3.8.1.windows-amd64/prometheus-3.8.1.windows-amd64"

# Start Prometheus (ONE LINE!)
./prometheus --config.file="/c/Users/Ari Azis/Downloads/SMSML_MohammadAri/Monitoring dan Logging/2.prometheus.yml"
```

✅ **Check:** You see "Server is ready to receive web requests"

---

**Terminal 2 - Start Inference API:**

Open **NEW Git Bash** terminal:

```bash
# Go to Monitoring folder
cd "/c/Users/Ari Azis/Downloads/SMSML_MohammadAri/Monitoring dan Logging"

# Activate venv
source "/c/Users/Ari Azis/Downloads/Submisi asah sistem ML/Eksperimen_SML_MohammadAri/venv/Scripts/activate"

# Install prometheus-client (if not already installed)
pip install prometheus-client flask psutil

# Start API
python 7.Inference.py
```

✅ **Check:** You see "Running on http://127.0.0.1:5001" and "Model and scaler loaded successfully"

**Screenshot:** Take screenshot of this terminal → save as `1.bukti_serving.jpg` in `Monitoring dan Logging` folder

---

**Terminal 3 - Generate Test Data:**

Open **NEW Git Bash** terminal (3rd one!):

```bash
# Go to Monitoring folder
cd "/c/Users/Ari Azis/Downloads/SMSML_MohammadAri/Monitoring dan Logging"

# Activate venv
source "/c/Users/Ari Azis/Downloads/Submisi asah sistem ML/Eksperimen_SML_MohammadAri/venv/Scripts/activate"

# Generate 100 test predictions
python test_predictions.py
```

⏰ **Wait** for 100 predictions to complete (takes ~1 minute).

---

### 3E. Verify Prometheus is Working

1. Open browser: **http://localhost:9090/targets**
2. Look for "fraud-detection-app"
3. **State should be: UP (green)** ✅

If it shows DOWN:
- Double-check you fixed the port in `2.prometheus.yml` (step 3B)
- Make sure the Inference API is running (Terminal 2)
- Restart Prometheus

---

### 3F. Take Prometheus Screenshots (5 screenshots)

1. Open browser: **http://localhost:9090**
2. Click on "Graph" tab at the top
3. Create folder: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Monitoring dan Logging\4.bukti monitoring Prometheus\`

For each query below:
- Type the query in the query box
- Click "Execute"
- Click "Graph" tab to see the chart
- Take screenshot
- Save with the specified name in the `4.bukti monitoring Prometheus` folder

**Query 1:**
```
fraud_predictions_total
```
- Save as: `1.monitoring_predictions_total.jpg`

**Query 2:** (Use this special formula for latency)
```
rate(fraud_prediction_latency_seconds_sum[5m]) / rate(fraud_prediction_latency_seconds_count[5m])
```
- Save as: `2.monitoring_latency.jpg`

**Query 3:**
```
fraud_rate_current
```
- Save as: `3.monitoring_fraud_rate.jpg`

**Query 4:**
```
active_requests
```
- Save as: `4.monitoring_active_requests.jpg`

**Query 5:**
```
rate(process_cpu_seconds_total[5m])
```
- Save as: `5.monitoring_system_resources.jpg`

✅ **You should now have 5 screenshots in the `4.bukti monitoring Prometheus` folder**

---

### 3G. Take Grafana Screenshots (5 screenshots)

1. Open browser: **http://localhost:3000**
2. Login: username `admin`, password `admin` (change password when prompted)
3. Create folder: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Monitoring dan Logging\5.bukti monitoring Grafana\`

**Add Prometheus Data Source:**
- Click ⚙️ (Configuration) → Data Sources
- Click "Add data source"
- Select "Prometheus"
- URL: `http://localhost:9090`
- Click "Save & Test" (should say "Data source is working")

**Create Dashboard:**
- Click + → Dashboard → Add new panel

**For each query, create a panel:**

1. **Panel 1:** Total Predictions
   - Query: `fraud_predictions_total`
   - Panel title: "Total Fraud Predictions"
   - Click "Apply"
   - Take screenshot → save as `1.monitoring_predictions.jpg`

2. **Panel 2:** Average Latency
   - Add new panel (click "Add panel" button)
   - Query: `rate(fraud_prediction_latency_seconds_sum[5m]) / rate(fraud_prediction_latency_seconds_count[5m])`
   - Panel title: "Average Prediction Latency"
   - Click "Apply"
   - Screenshot → `2.monitoring_latency.jpg`

3. **Panel 3:** Fraud Rate
   - Add new panel
   - Query: `fraud_rate_current`
   - Panel title: "Current Fraud Rate"
   - Click "Apply"
   - Screenshot → `3.monitoring_fraud_rate.jpg`

4. **Panel 4:** Active Requests
   - Add new panel
   - Query: `active_requests`
   - Panel title: "Active Requests"
   - Click "Apply"
   - Screenshot → `4.monitoring_active_requests.jpg`

5. **Panel 5:** CPU Usage
   - Add new panel
   - Query: `rate(process_cpu_seconds_total[5m])`
   - Panel title: "CPU Usage Rate"
   - Click "Apply"
   - Screenshot → `5.monitoring_system.jpg`

**IMPORTANT:** Save dashboard as "Ethereum Fraud Detection - YourDicodingUsername"
- Click 💾 (Save dashboard) icon at the top
- Name: "Ethereum Fraud Detection - [Your Dicoding Username]"
- Click "Save"

✅ **You should now have 5 screenshots in the `5.bukti monitoring Grafana` folder**

---

### 3H. Take Grafana Alert Screenshots (2 screenshots)

1. Create folder: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Monitoring dan Logging\6.bukti alerting Grafana\`

2. In Grafana:
   - Click 🔔 (Alerting) → Alert rules
   - Click "New alert rule"

3. Configure the alert:
   - **Rule name:** "High Fraud Rate Alert"
   - **Section A:**
     - Select your Prometheus data source
     - Query: `fraud_rate_current`
   - **Section B (Condition):**
     - Set threshold: `WHEN last() OF A IS ABOVE 30`
   - **Section C (Alert details):**
     - Folder: "General Alerting"
     - Evaluation group: Create new "fraud-alerts"
     - Evaluation interval: 1m
   - Click "Save rule"

**Screenshot 1:** Take screenshot of the alert rule configuration page → save as `1.rules_high_fraud_rate.jpg`

**Screenshot 2:**
- Go back to Alert rules list (🔔 → Alert rules)
- You should see your "High Fraud Rate Alert" listed
- Take screenshot → save as `2.notifikasi_high_fraud_rate.jpg`

✅ **Done with Kriteria 4!**

---

## STEP 4: Verify GitHub Actions (5 minutes)

1. Go to: https://github.com/13222093/Workflow-CI
2. Click "Actions" tab
3. Click "Run workflow" → Select branch: `main` → Click "Run workflow" (green button)
4. Wait ~5 minutes for it to complete
5. Should show green checkmark ✅

✅ **Done with Kriteria 3!**

---

## STEP 5: Final Check (5 minutes)

Your `SMSML_MohammadAri` folder should contain:

```
SMSML_MohammadAri/
├── Eksperimen_SML_MohammadAri.txt          ✅ (with your username: 13222093)
├── Workflow-CI.txt                          ✅ (with your username: 13222093)
├── Membangun_model/
│   ├── screenshoot_dashboard.jpg            📸 NEW (Step 2D)
│   ├── screenshoot_artifak.jpg              📸 NEW (Step 2D)
│   ├── modelling_tuning.py                  ✅ (fixed XGBoost bug)
│   └── ... (other files)
└── Monitoring dan Logging/
    ├── 1.bukti_serving.jpg                  📸 NEW (Step 3D)
    ├── 2.prometheus.yml                      ✅ (fixed port to 5001)
    ├── best_xgboost.pkl                      ✅ NEW (copied in Step 3C)
    ├── scaler_model.pkl                      ✅ NEW (copied in Step 3C)
    ├── 4.bukti monitoring Prometheus/        📁 5 screenshots (Step 3F)
    │   ├── 1.monitoring_predictions_total.jpg
    │   ├── 2.monitoring_latency.jpg
    │   ├── 3.monitoring_fraud_rate.jpg
    │   ├── 4.monitoring_active_requests.jpg
    │   └── 5.monitoring_system_resources.jpg
    ├── 5.bukti monitoring Grafana/           📁 5 screenshots (Step 3G)
    │   ├── 1.monitoring_predictions.jpg
    │   ├── 2.monitoring_latency.jpg
    │   ├── 3.monitoring_fraud_rate.jpg
    │   ├── 4.monitoring_active_requests.jpg
    │   └── 5.monitoring_system.jpg
    └── 6.bukti alerting Grafana/             📁 2 screenshots (Step 3H)
        ├── 1.rules_high_fraud_rate.jpg
        └── 2.notifikasi_high_fraud_rate.jpg
```

**Total screenshots:** 2 + 1 + 5 + 5 + 2 = **15 screenshots**

---

## STEP 6: Submit (5 minutes)

1. Close all terminals (Ctrl+C to stop services)
2. Right-click `SMSML_MohammadAri` folder
3. Send to → Compressed (zipped) folder
4. You'll get: `SMSML_MohammadAri.zip`
5. Upload to your submission platform

✅ **DONE!** 🎉

---

## TROUBLESHOOTING

### "Command not found" error?
- Make sure you activated the virtual environment (`source venv/Scripts/activate`)
- You should see `(venv)` at the start of your prompt

### "Empty query result" in Prometheus?
- Check http://localhost:9090/targets - should show "UP"
- Make sure you fixed the port in `2.prometheus.yml` (Step 3B)
- Wait 30 seconds after running test_predictions.py for metrics to appear
- Verify metrics exist at: http://localhost:5001/metrics

### "Model not found" error in Inference API?
- You need to copy model files from Membangun_model (Step 3C)
- Make sure `best_xgboost.pkl` and `scaler_model.pkl` are in `Monitoring dan Logging/`

### Git push failed?
- Make sure you created the GitHub repositories first (https://github.com/new)
- Repository names MUST be: `Eksperimen_SML_MohammadAri` and `Workflow-CI`
- Make sure repos are PUBLIC

### XGBoost TypeError?
- You forgot to fix line 304 in `modelling_tuning.py` (Step 2A)
- Change `mlflow.xgboost.log_model` to `mlflow.sklearn.log_model`

### Prometheus target shows DOWN?
- Port mismatch! Check `2.prometheus.yml` uses `localhost:5001` not `8000`
- Restart Prometheus after fixing the config

### No prometheus-client error?
- Run: `pip install prometheus-client flask psutil`

---

## Summary Timeline

- **Step 1:** Push to GitHub - 15 min
- **Step 2:** MLflow screenshots - 30 min
- **Step 3:** Monitoring screenshots - 60 min
- **Step 4:** Verify GitHub Actions - 5 min
- **Step 5:** Final check - 5 min
- **Step 6:** Submit - 5 min

**Total:** ~2 hours

---

## Key Files Reference

**Files you edited:**
1. `modelling_tuning.py` - Line 304 (XGBoost fix)
2. `2.prometheus.yml` - Port number (8000 → 5001)

**Files you copied:**
1. `best_xgboost.pkl` - From Membangun_model to Monitoring dan Logging
2. `scaler_model.pkl` - From Membangun_model to Monitoring dan Logging

**GitHub Repos:**
1. https://github.com/13222093/Eksperimen_SML_MohammadAri
2. https://github.com/13222093/Workflow-CI

---

Good luck with your submission! 🚀

**Having issues?** Re-read the specific step or check the TROUBLESHOOTING section above.
