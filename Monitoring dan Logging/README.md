# Monitoring dan Logging - Kriteria 4

## Overview
This folder contains all files and configurations for monitoring and logging the Ethereum Fraud Detection model using Prometheus and Grafana.

## Kriteria 4 Levels

### Basic (2 pts)
- ✅ Model serving (MLflow or Flask API)
- ✅ Prometheus monitoring with 3 different metrics
- ✅ Grafana visualization with same metrics as Prometheus

### Skilled (3 pts)
- ✅ Basic requirements met
- ✅ Grafana monitoring with 5 different metrics
- ✅ 1 alert rule in Grafana

### Advanced (4 pts)
- Basic & Skilled requirements met
- 10 different metrics in Grafana
- 3 alert rules in Grafana

## Project Structure

```
Monitoring dan Logging/
├── 1.bukti_serving.txt                    # Instructions for serving screenshot
├── 2.prometheus.yml                        # Prometheus configuration
├── 3.prometheus_exporter.py                # Custom metrics exporter
├── 4.bukti monitoring Prometheus/          # Prometheus screenshots folder
│   ├── INSTRUCTIONS.md
│   ├── 1.monitoring_predictions_total.jpg (YOU CREATE THIS)
│   ├── 2.monitoring_latency.jpg (YOU CREATE THIS)
│   ├── 3.monitoring_fraud_rate.jpg (YOU CREATE THIS)
│   ├── 4.monitoring_active_requests.jpg (YOU CREATE THIS)
│   └── 5.monitoring_system_resources.jpg (YOU CREATE THIS)
├── 5.bukti monitoring Grafana/             # Grafana screenshots folder
│   ├── INSTRUCTIONS.md
│   ├── 1.monitoring_predictions.jpg (YOU CREATE THIS)
│   ├── 2.monitoring_latency.jpg (YOU CREATE THIS)
│   ├── 3.monitoring_fraud_rate.jpg (YOU CREATE THIS)
│   ├── 4.monitoring_active_requests.jpg (YOU CREATE THIS)
│   └── 5.monitoring_system.jpg (YOU CREATE THIS)
├── 6.bukti alerting Grafana/               # Grafana alerts screenshots
│   ├── INSTRUCTIONS.md
│   ├── 1.rules_high_fraud_rate.jpg (YOU CREATE THIS)
│   └── 2.notifikasi_high_fraud_rate.jpg (YOU CREATE THIS)
├── 7.inference.py                          # Flask inference API with metrics
└── README.md                               # This file
```

## Prerequisites

Install required packages:
```bash
pip install prometheus-client flask psutil
```

For Prometheus and Grafana, download and install:
- **Prometheus**: https://prometheus.io/download/
- **Grafana**: https://grafana.com/grafana/download

## Setup Guide

### Step 1: Start Model Serving

Choose ONE of the following options:

#### Option A: MLflow Model Serve (Simple)
```bash
# Get RUN_ID from MLflow UI (http://localhost:5000)
mlflow ui  # First, start MLflow UI to find your run ID

# Then serve the model
mlflow models serve -m runs:/<RUN_ID>/xgboost_model -p 5000 --no-conda
```

#### Option B: Flask Inference API (Recommended - includes metrics)
```bash
# Copy model files to Monitoring dan Logging folder
cp ../Membangun_model/best_xgboost.pkl .
cp ../Membangun_model/scaler_model.pkl .

# Run the inference API
python 7.inference.py
```

The API will be available at:
- http://localhost:5001/ - API info
- http://localhost:5001/health - Health check
- http://localhost:5001/predict - Prediction endpoint
- http://localhost:5001/metrics - Prometheus metrics

**Take screenshot for**: `1.bukti_serving.jpg`

### Step 2: Start Prometheus Metrics Exporter (Optional but Recommended)

If using MLflow serve (Option A), run the custom exporter:
```bash
python 3.prometheus_exporter.py
```

This will expose metrics at http://localhost:8000/metrics

(If using Flask API (Option B), metrics are already included!)

### Step 3: Configure and Start Prometheus

1. **Edit prometheus.yml** if needed (already configured)

2. **Start Prometheus**:
   ```bash
   # On Windows
   prometheus.exe --config.file=2.prometheus.yml

   # On Linux/Mac
   ./prometheus --config.file=2.prometheus.yml
   ```

3. **Access Prometheus UI**: http://localhost:9090

4. **Verify metrics are being scraped**:
   - Go to Status → Targets
   - Should see your endpoints as "UP"

5. **Take screenshots** (see instructions in `4.bukti monitoring Prometheus/INSTRUCTIONS.md`):
   - `1.monitoring_predictions_total.jpg`
   - `2.monitoring_latency.jpg`
   - `3.monitoring_fraud_rate.jpg`
   - `4.monitoring_active_requests.jpg`
   - `5.monitoring_system_resources.jpg`

### Step 4: Setup Grafana

1. **Start Grafana**:
   ```bash
   # On Windows (if installed as service, it auto-starts)
   # Otherwise:
   grafana-server.exe

   # On Linux/Mac
   ./grafana-server
   ```

2. **Access Grafana**: http://localhost:3000
   - Default credentials: admin/admin
   - Change password when prompted

3. **Add Prometheus Data Source**:
   - Configuration (⚙️) → Data Sources → Add data source
   - Select "Prometheus"
   - URL: `http://localhost:9090`
   - Click "Save & Test"

4. **Create Dashboard**:
   - Click "+" → Dashboard → Add new panel
   - Create panels for each metric (see `5.bukti monitoring Grafana/INSTRUCTIONS.md`)

5. **IMPORTANT**: Name your dashboard with your name!
   - Example: "Ethereum Fraud Detection - Mohammad Ari"

6. **Take screenshots** (minimum 5 for Skilled):
   - `1.monitoring_predictions.jpg`
   - `2.monitoring_latency.jpg`
   - `3.monitoring_fraud_rate.jpg`
   - `4.monitoring_active_requests.jpg`
   - `5.monitoring_system.jpg`

### Step 5: Create Grafana Alerts (Skilled Level)

1. **Create Alert Rule**:
   - Alerting → Alert rules → New alert rule
   - Follow instructions in `6.bukti alerting Grafana/INSTRUCTIONS.md`

2. **Recommended first alert**: High Fraud Rate
   - Metric: `fraud_rate_current`
   - Condition: `> 30`

3. **Test the alert**:
   - Use "Test" button in alert rule
   - Or trigger naturally by generating predictions

4. **Take screenshots** (minimum 1 alert for Skilled):
   - `1.rules_high_fraud_rate.jpg` - Alert rule configuration
   - `2.notifikasi_high_fraud_rate.jpg` - Alert notification

## Testing the System

### Generate Test Traffic

Use this Python script to generate predictions and trigger metrics:

```python
import requests
import random
import time

url = "http://localhost:5001/predict"

for i in range(100):
    # Random feature values (16 features)
    features = [random.uniform(0, 1) for _ in range(16)]

    response = requests.post(url, json={"features": features})
    print(f"Prediction {i+1}: {response.json()}")

    time.sleep(0.5)  # Wait 0.5 seconds between requests
```

Save as `test_predictions.py` and run:
```bash
python test_predictions.py
```

This will:
- Generate prediction traffic
- Populate metrics in Prometheus
- Create data for Grafana dashboards
- Potentially trigger alerts if thresholds are met

## Key Metrics Available

### Application Metrics (from inference.py or exporter)

1. **fraud_predictions_total** - Total predictions by type (fraud/non-fraud)
2. **fraud_prediction_latency_seconds** - Prediction latency distribution
3. **fraud_rate_current** - Current fraud detection rate
4. **active_requests** - Number of active prediction requests
5. **fraud_detection_errors_total** - Error counts by type

### System Metrics

6. **process_cpu_seconds_total** - CPU usage
7. **process_resident_memory_bytes** - Memory usage
8. **up** - Service availability

### Model Performance (from exporter)

9. **fraud_detection_model_accuracy** - Model accuracy
10. **fraud_detection_model_precision** - Model precision
11. **fraud_detection_model_recall** - Model recall
12. **fraud_detection_model_f1_score** - F1 score

## Checklist for Submission

### For Skilled Level (3 pts):

- [ ] `1.bukti_serving.jpg` - Screenshot showing model serving
- [ ] `2.prometheus.yml` - Prometheus configuration file ✅
- [ ] `3.prometheus_exporter.py` - Metrics exporter script ✅
- [ ] `4.bukti monitoring Prometheus/` folder with 5 screenshots:
  - [ ] `1.monitoring_predictions_total.jpg`
  - [ ] `2.monitoring_latency.jpg`
  - [ ] `3.monitoring_fraud_rate.jpg`
  - [ ] `4.monitoring_active_requests.jpg`
  - [ ] `5.monitoring_system_resources.jpg`
- [ ] `5.bukti monitoring Grafana/` folder with 5 screenshots (same metrics):
  - [ ] `1.monitoring_predictions.jpg`
  - [ ] `2.monitoring_latency.jpg`
  - [ ] `3.monitoring_fraud_rate.jpg`
  - [ ] `4.monitoring_active_requests.jpg`
  - [ ] `5.monitoring_system.jpg`
- [ ] `6.bukti alerting Grafana/` folder with 2 screenshots (1 alert):
  - [ ] `1.rules_high_fraud_rate.jpg`
  - [ ] `2.notifikasi_high_fraud_rate.jpg`
- [ ] `7.inference.py` - Inference API script ✅

## Troubleshooting

### Problem: Prometheus not scraping metrics
- **Solution**: Check if your endpoints are running and accessible
- Verify targets in Prometheus UI (Status → Targets)
- Check firewall settings

### Problem: Grafana shows "No data"
- **Solution**:
  - Verify Prometheus data source is connected
  - Check time range (use "Last 5 minutes")
  - Generate some predictions to create data
  - Verify metrics exist in Prometheus first

### Problem: Alerts not triggering
- **Solution**:
  - Lower the threshold temporarily
  - Use "Test" button in alert rule
  - Check evaluation interval
  - Verify query returns data

### Problem: Port already in use
- **Solution**:
  - Change port in the script
  - Kill process using the port: `netstat -ano | findstr :5001`
  - Use different ports for each service

## Notes

- Ensure all screenshots include your name (in dashboard title or window)
- Screenshots should show timestamps
- Use descriptive dashboard and alert names
- Keep services running while taking screenshots
- Test everything before final submission

## Additional Resources

- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
- MLflow Model Serving: https://mlflow.org/docs/latest/models.html#deploy-mlflow-models

---

**Author**: Mohammad Ari Alexander Aziz
**Date**: December 2024
**Purpose**: Kriteria 4 - Monitoring dan Logging submission for Ethereum Fraud Detection project
