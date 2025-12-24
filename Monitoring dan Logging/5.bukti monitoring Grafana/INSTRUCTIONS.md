# Bukti Monitoring Grafana

## Untuk Skilled Level: Minimal 5 Metrics (sama seperti Prometheus)

Ambil screenshot dari Grafana dashboard (http://localhost:3000) untuk minimal 5 metrics yang sama dengan Prometheus.

### Setup Dashboard di Grafana:

1. Login ke Grafana (default: admin/admin)
2. Add Prometheus data source:
   - Configuration → Data Sources → Add data source → Prometheus
   - URL: http://localhost:9090
   - Save & Test
3. Create New Dashboard
4. Add panels untuk setiap metric

### Panels yang Harus Dibuat:

1. **Total Predictions**
   - Panel type: Time series atau Stat
   - Query: `fraud_predictions_total`
   - Simpan sebagai: `1.monitoring_predictions.jpg`

2. **Prediction Latency**
   - Panel type: Time series
   - Query: `rate(fraud_prediction_latency_seconds_sum[5m]) / rate(fraud_prediction_latency_seconds_count[5m])`
   - Simpan sebagai: `2.monitoring_latency.jpg`

3. **Fraud Rate**
   - Panel type: Gauge atau Stat
   - Query: `fraud_rate_current`
   - Simpan sebagai: `3.monitoring_fraud_rate.jpg`

4. **Active Requests**
   - Panel type: Time series
   - Query: `active_requests`
   - Simpan sebagai: `4.monitoring_active_requests.jpg`

5. **System Resources**
   - Panel type: Time series (multi-series)
   - Queries: CPU, Memory metrics
   - Simpan sebagai: `5.monitoring_system.jpg`

## Tips untuk Dashboard:

- Beri nama dashboard: "Ethereum Fraud Detection - [Nama Anda]"
- Set time range: Last 5 minutes atau Last 15 minutes
- Gunakan refresh rate: 5s atau 10s
- Pastikan semua panels menampilkan data

## Cara Mengambil Screenshot:

1. Buka Grafana di http://localhost:3000
2. Pastikan dashboard sudah dibuat dengan semua panels
3. Klik masing-masing panel untuk fokus (atau screenshot full dashboard)
4. Screenshot harus mencakup:
   - Nama dashboard (harus termasuk nama Anda!)
   - Panel dengan data
   - Timestamp/time range
   - Legend (jika ada multiple series)

## Format Screenshot:
- Format: JPG atau PNG
- Nama file: `X.monitoring_<nama_metric>.jpg`
- Bisa screenshot per-panel atau full dashboard
