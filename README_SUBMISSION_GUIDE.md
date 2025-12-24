# Submission Guide - SMSML Mohammad Ari Alexander Aziz

## ✅ What's Ready

Your project has been restructured to match the official submission format!

### Current Structure:

```
C:\Users\Ari Azis\Downloads\
├── SMSML_MohammadAri/                          ← FINAL SUBMISSION FOLDER
│   ├── Eksperimen_SML_MohammadAri.txt          ← Template (update with GitHub URL)
│   ├── Membangun_model/                         ← Ready ✓
│   ├── Workflow-CI.txt                          ← Template (update with GitHub URL)
│   └── Monitoring dan Logging/                  ← Ready ✓
│
└── GitHub_Repos_ToUpload/                       ← STAGING AREA FOR GITHUB
    ├── Eksperimen_SML_MohammadAri/              ← Push this to GitHub
    └── Workflow-CI/                             ← Push this to GitHub
```

---

## 📋 Next Steps (In Order)

### Step 1: Create GitHub Repositories

1. Go to https://github.com/new
2. Create **2 public repositories**:
   - Repository 1: `Eksperimen_SML_MohammadAri`
   - Repository 2: `Workflow-CI`
3. **IMPORTANT**: Set visibility to **Public**

### Step 2: Push to GitHub

#### For Repository 1 (Eksperimen_SML_MohammadAri):

```bash
cd "C:\Users\Ari Azis\Downloads\GitHub_Repos_ToUpload\Eksperimen_SML_MohammadAri"
git init
git add .
git commit -m "Initial commit: Ethereum fraud detection preprocessing"
git branch -M main
git remote add origin https://github.com/[YOUR_USERNAME]/Eksperimen_SML_MohammadAri.git
git push -u origin main
```

#### For Repository 2 (Workflow-CI):

```bash
cd "C:\Users\Ari Azis\Downloads\GitHub_Repos_ToUpload\Workflow-CI"
git init
git add .
git commit -m "Initial commit: MLflow Project with CI/CD"
git branch -M main
git remote add origin https://github.com/[YOUR_USERNAME]/Workflow-CI.git
git push -u origin main
```

**Replace `[YOUR_USERNAME]` with your actual GitHub username!**

### Step 3: Update .txt Files

After pushing to GitHub, update these files in `SMSML_MohammadAri/`:

1. Open `Eksperimen_SML_MohammadAri.txt`
2. Replace `[YOUR_USERNAME]` with your actual GitHub username
3. Open `Workflow-CI.txt`
4. Replace `[YOUR_USERNAME]` with your actual GitHub username

### Step 4: Run Scripts & Take Screenshots

#### For Kriteria 2 (2 screenshots needed):

**Important**: Use Git Bash or PowerShell, NOT Command Prompt for best compatibility.

##### Option 1: Using PowerShell (Recommended)

```powershell
# Navigate to Membangun_model
cd "C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Membangun_model"

# Activate venv from the old location
C:\Users\Ari Azis\Downloads\Submisi asah sistem ML\Eksperimen_SML_MohammadAri\venv\Scripts\Activate.ps1

# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Run training (this will take 5-15 minutes)
python modelling_tuning.py

# After training completes, start MLflow UI (in the same or new terminal)
mlflow ui
```

##### Option 2: Using Git Bash

```bash
# Navigate to Membangun_model
cd "C:/Users/Ari Azis/Downloads/SMSML_MohammadAri/Membangun_model"

# Activate venv
source "C:/Users/Ari Azis/Downloads/Submisi asah sistem ML/Eksperimen_SML_MohammadAri/venv/Scripts/activate"

# Run training
python modelling_tuning.py

# Start MLflow UI
mlflow ui
```

**Then take screenshots:**
1. Open browser: http://localhost:5000
2. You'll see the MLflow UI with experiment runs
3. Take screenshot: `screenshoot_dashboard.jpg` (show the runs list with metrics)
4. Click on one run → Click "Artifacts" tab
5. Take screenshot: `screenshoot_artifak.jpg` (show model artifacts, plots, etc.)

**Save both screenshots in**: `C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Membangun_model\`

---

#### For Kriteria 4 (12 screenshots needed):

##### Step 4.1: Install Required Tools First

**Install Prometheus:**
1. Download from: https://prometheus.io/download/
   - Get: `prometheus-X.X.X.windows-amd64.zip` (latest version)
2. Extract to: `C:\Tools\prometheus\` (or any location you prefer)
3. Remember this path for later!

**Install Grafana:**
1. Download from: https://grafana.com/grafana/download?platform=windows
   - Get the Windows installer (.msi or .zip)
2. Install or extract to: `C:\Tools\grafana\` (or any location)
3. Grafana will run as a service after installation

##### Step 4.2: Start Services (Need 3 Terminals!)

**Terminal 1 - Inference API:**

```powershell
# PowerShell
cd "C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Monitoring dan Logging"

# Activate venv (reuse the one from Kriteria 1)
C:\Users\Ari Azis\Downloads\Submisi asah sistem ML\Eksperimen_SML_MohammadAri\venv\Scripts\Activate.ps1

# Install additional dependencies if needed
pip install prometheus-client flask psutil

# Start the API
python 7.Inference.py
```

**Terminal 2 - Prometheus:**

```powershell
# PowerShell - navigate to where you extracted Prometheus
cd C:\Tools\prometheus

# Start Prometheus with our config
.\prometheus.exe --config.file="C:\Users\Ari Azis\Downloads\SMSML_MohammadAri\Monitoring dan Logging\2.prometheus.yml"
```

**Terminal 3 - Grafana:**

If installed as Windows service (installer):
```powershell
# Check if running
Get-Service -Name "Grafana"

# Start if not running
Start-Service -Name "Grafana"
```

If using standalone (zip):
```powershell
cd C:\Tools\grafana\bin
.\grafana-server.exe
```

##### Step 4.3: Access UIs and Take Screenshots

**A) Inference API** (Terminal output):
- API running at: http://localhost:5001
- Test it: Open browser → http://localhost:5001/health
- Take screenshot: `1.bukti_serving.jpg`
- Should show JSON response with service status

**B) Prometheus UI** (5 screenshots):
1. Open: http://localhost:9090
2. Go to "Status" → "Targets" to verify metrics are being scraped
3. Go to "Graph" tab
4. Enter queries and take screenshots:
   - `fraud_predictions_total` → Save as `1.monitoring_predictions_total.jpg`
   - `fraud_prediction_latency_seconds` → Save as `2.monitoring_latency.jpg`
   - `fraud_rate_current` → Save as `3.monitoring_fraud_rate.jpg`
   - `active_requests` → Save as `4.monitoring_active_requests.jpg`
   - `process_cpu_seconds_total` → Save as `5.monitoring_system_resources.jpg`
5. Save all in: `4.bukti monitoring Prometheus/`

**C) Grafana UI** (5 screenshots):
1. Open: http://localhost:3000
2. Login: admin / admin (change password when prompted)
3. Add Prometheus data source:
   - Configuration → Data Sources → Add data source
   - Select "Prometheus"
   - URL: `http://localhost:9090`
   - Click "Save & Test"
4. Create Dashboard:
   - "+" → Dashboard → Add new panel
   - Add 5 panels for the same metrics as Prometheus
   - **IMPORTANT**: Name dashboard: "Ethereum Fraud Detection - [Your Dicoding Username]"
5. Take screenshots of each panel:
   - Save as: `1.monitoring_predictions.jpg` through `5.monitoring_system.jpg`
   - Save all in: `5.bukti monitoring Grafana/`

**D) Grafana Alerts** (2 screenshots):
1. In Grafana: Alerting → Alert rules → New alert rule
2. Create alert: "High Fraud Rate"
   - Query: `fraud_rate_current`
   - Condition: `WHEN last() OF A IS ABOVE 30`
   - Name: "High Fraud Rate Alert"
3. Take screenshot of alert rule config: `1.rules_high_fraud_rate.jpg`
4. Test the alert or wait for it to trigger
5. Take screenshot of notification: `2.notifikasi_high_fraud_rate.jpg`
6. Save both in: `6.bukti alerting Grafana/`

##### Step 4.4: Generate Test Data (Important!)

To populate metrics for screenshots, generate some predictions:

**Create test script** (`test_predictions.py` in Monitoring dan Logging folder):
```python
import requests
import random
import time

url = "http://localhost:5001/predict"

for i in range(100):
    # Random feature values (16 features)
    features = [random.uniform(0, 1) for _ in range(16)]

    try:
        response = requests.post(url, json={"features": features})
        print(f"Prediction {i+1}: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(0.5)
```

Then run:
```powershell
python test_predictions.py
```

This generates traffic so Prometheus/Grafana have data to display!

### Step 5: Verify GitHub Actions (Workflow-CI)

1. Go to your GitHub repo: `Workflow-CI`
2. Click "Actions" tab
3. Manually trigger the workflow or push a small change
4. Ensure workflow runs successfully (green checkmark)
5. This proves your CI/CD is working!

### Step 6: Final Verification

Check that `SMSML_MohammadAri/` contains:

- ✅ `Eksperimen_SML_MohammadAri.txt` (with real GitHub URL)
- ✅ `Workflow-CI.txt` (with real GitHub URL)
- ✅ `Membangun_model/`
  - ✅ `modelling.py`
  - ✅ `modelling_tuning.py`
  - ✅ `ethereum_fraud_preprocessing.csv`
  - 📸 `screenshoot_dashboard.jpg`
  - 📸 `screenshoot_artifak.jpg`
  - ✅ `requirements.txt`
- ✅ `Monitoring dan Logging/`
  - 📸 `1.bukti_serving.jpg`
  - ✅ `2.prometheus.yml`
  - ✅ `3.prometheus_exporter.py`
  - 📸 `4.bukti monitoring Prometheus/` (5 screenshots)
  - 📸 `5.bukti monitoring Grafana/` (5 screenshots)
  - 📸 `6.bukti alerting Grafana/` (2 screenshots)
  - ✅ `7.Inference.py`

### Step 7: Submit!

1. Compress `SMSML_MohammadAri/` folder to `.zip`
2. Upload to submission platform
3. Double-check GitHub repos are Public

---

## 🔧 Key Changes Made

1. ✅ Updated MLflow version: `>=2.9.0` → `==2.19.0`
2. ✅ Updated Python version: `3.10` → `3.12.7` (in conda.yaml)
3. ✅ Renamed: `7.inference.py` → `7.Inference.py`
4. ✅ Created proper submission structure
5. ✅ Prepared GitHub repo staging folders
6. ✅ Created `.txt` template files for GitHub links

---

## ⚠️ Important Reminders

- Both GitHub repos MUST be **Public**
- GitHub Actions workflow must run successfully at least once
- All notebook cells must run without errors
- Screenshots must include your Dicoding username in dashboard names
- `.txt` files must contain actual GitHub URLs (not `[YOUR_USERNAME]`)

---

## 🎯 Expected Score

Targeting **Skilled Level (3 points × 4 criteria = 12 points)**:

| Kriteria | Status | Evidence |
|----------|--------|----------|
| 1. Preprocessing | ✅ Ready | GitHub repo + automation script |
| 2. Model Building | ⏸️ Need screenshots | modelling_tuning.py ready |
| 3. CI Workflow | ✅ Ready | GitHub repo + Actions |
| 4. Monitoring | ⏸️ Need screenshots | All scripts ready |

---

## 📞 If You Have Issues

**PowerShell activation error?**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Git not installed?**
Download from: https://git-scm.com/download/win

**GitHub authentication?**
Use Personal Access Token or GitHub CLI

---

Good luck with your submission! 🚀

**Remember**: All the hard work is done. You just need to:
1. Push to GitHub (5 min)
2. Update .txt files (1 min)
3. Take screenshots (30-60 min)
4. Submit! ✅
