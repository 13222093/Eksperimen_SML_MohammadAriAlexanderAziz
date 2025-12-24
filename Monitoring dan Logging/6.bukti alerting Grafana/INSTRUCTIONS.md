# Bukti Alerting Grafana

## Untuk Skilled Level: Minimal 1 Alert

Buat minimal 1 alert rule di Grafana dan ambil screenshot untuk:
- Alert rule configuration
- Alert notification (bisa simulated/test)

### Alert yang Disarankan untuk Fraud Detection:

#### Alert 1: High Fraud Rate (WAJIB untuk Skilled)
- **Metric**: `fraud_rate_current`
- **Condition**: `fraud_rate_current > 30`
- **Severity**: Warning
- **Description**: "Fraud rate melebihi 30%, perlu investigasi!"

Screenshots yang diperlukan:
1. `1.rules_high_fraud_rate.jpg` - Screenshot alert rule configuration
2. `2.notifikasi_high_fraud_rate.jpg` - Screenshot alert notification/trigger

---

## Cara Setup Alert di Grafana:

### 1. Buat Alert Rule

1. **Navigasi**:
   - Grafana → Alerting → Alert rules → New alert rule

2. **Set Query and Conditions**:
   - Query A: `fraud_rate_current`
   - Condition: `WHEN last() OF A IS ABOVE 30`
   - For: 1m (alert setelah condition terpenuhi selama 1 menit)

3. **Alert Details**:
   - Rule name: `High Fraud Rate Alert`
   - Folder: Default atau buat folder baru
   - Evaluation group: Create new (interval: 1m)

4. **Annotations**:
   - Summary: `High fraud rate detected: {{ $values.A }}%`
   - Description: `The current fraud rate is {{ $values.A }}%, which exceeds the threshold of 30%`

5. **Save** alert rule

### 2. Setup Notification Channel (Optional untuk trigger)

1. **Navigasi**:
   - Grafana → Alerting → Contact points

2. **Add Contact Point**:
   - Name: `Email Alert` atau `Slack Alert`
   - Type: Email (test dengan email dummy juga OK)
   - Email addresses: your-email@example.com
   - Save

3. **Create Notification Policy**:
   - Link alert rule dengan contact point yang dibuat

### 3. Test Alert

Ada 2 cara untuk test:

#### Cara 1: Trigger Naturally
- Jalankan inference.py
- Generate prediksi dengan fraud rate tinggi
- Tunggu alert terpantik secara natural

#### Cara 2: Simulate/Test (Lebih Mudah)
- Pada alert rule, klik tombol "Test"
- Grafana akan simulate condition
- Screenshot test result

---

## Screenshots yang Diperlukan:

### Screenshot 1: Alert Rule (`1.rules_<alert_name>.jpg`)

Harus menunjukkan:
- Alert rule name
- Query yang digunakan
- Condition threshold
- Evaluation interval
- Alert state (Pending/Firing/Normal)

**Contoh**: `1.rules_high_fraud_rate.jpg`

### Screenshot 2: Alert Notification (`2.notifikasi_<alert_name>.jpg`)

Harus menunjukkan:
- Alert firing/triggered
- Timestamp kapan alert trigger
- Alert message
- Notification status

**Contoh**: `2.notifikasi_high_fraud_rate.jpg`

Bisa berupa:
- Screenshot dari Grafana Alert History
- Screenshot email notification (jika setup email)
- Screenshot Slack/Discord notification (jika setup)
- Screenshot dari "Test Alert" result

---

## Tips:

1. **Untuk mempermudah trigger alert**:
   - Set threshold rendah (misal: fraud_rate > 10)
   - Atau test alert manually via Grafana UI

2. **Alert State Colors**:
   - 🟢 Normal - No alert
   - 🟡 Pending - Condition met, waiting for "For" duration
   - 🔴 Firing - Alert actively firing

3. **Pastikan Screenshots Jelas**:
   - Include nama Anda di dashboard/window title
   - Include timestamp
   - Include alert status yang jelas

4. **Format Screenshot**:
   - Resolution minimal: 1280x720
   - Format: JPG atau PNG
   - Tidak blur/terpotong

---

## Struktur File untuk Submission:

Untuk Skilled (1 alert):
```
6.bukti alerting Grafana/
├── 1.rules_high_fraud_rate.jpg
├── 2.notifikasi_high_fraud_rate.jpg
└── INSTRUCTIONS.md (file ini)
```

Untuk Advanced (3 alerts):
```
6.bukti alerting Grafana/
├── 1.rules_high_fraud_rate.jpg
├── 2.notifikasi_high_fraud_rate.jpg
├── 3.rules_high_latency.jpg
├── 4.notifikasi_high_latency.jpg
├── 5.rules_model_error.jpg
├── 6.notifikasi_model_error.jpg
└── INSTRUCTIONS.md
```

---

## Additional Alert Ideas (untuk Advanced - 3 alerts):

### Alert 2: High Prediction Latency
- **Metric**: `rate(fraud_prediction_latency_seconds_sum[5m]) / rate(fraud_prediction_latency_seconds_count[5m])`
- **Condition**: `> 0.5` (> 500ms average latency)
- **Severity**: Warning

### Alert 3: Model Error Rate
- **Metric**: `rate(fraud_detection_errors_total[5m])`
- **Condition**: `> 1` (more than 1 error per 5 minutes)
- **Severity**: Critical
