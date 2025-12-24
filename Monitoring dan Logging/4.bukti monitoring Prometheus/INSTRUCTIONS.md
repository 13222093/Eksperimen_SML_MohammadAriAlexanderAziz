# Bukti Monitoring Prometheus

## Untuk Skilled Level: Minimal 5 Metrics

Ambil screenshot dari Prometheus UI (http://localhost:9090) untuk minimal 5 metrics yang berbeda.

### Metrics yang Harus Di-Screenshot:

1. **fraud_predictions_total**
   - Query: `fraud_predictions_total`
   - Menunjukkan total prediksi fraud vs non-fraud
   - Simpan sebagai: `1.monitoring_predictions_total.jpg`

2. **fraud_prediction_latency_seconds**
   - Query: `fraud_prediction_latency_seconds`
   - Menunjukkan distribusi latency prediksi
   - Simpan sebagai: `2.monitoring_latency.jpg`

3. **fraud_rate_current**
   - Query: `fraud_rate_current`
   - Menunjukkan fraud rate saat ini
   - Simpan sebagai: `3.monitoring_fraud_rate.jpg`

4. **active_requests**
   - Query: `active_requests`
   - Menunjukkan jumlah request aktif
   - Simpan sebagai: `4.monitoring_active_requests.jpg`

5. **CPU/Memory Usage**
   - Query: `process_cpu_seconds_total` atau metric sistem lainnya
   - Simpan sebagai: `5.monitoring_system_resources.jpg`

## Cara Mengambil Screenshot:

1. Buka Prometheus UI di http://localhost:9090
2. Di tab "Graph", masukkan query metric
3. Klik "Execute"
4. Pilih tab "Graph" untuk melihat visualisasi
5. Screenshot harus mencakup:
   - Query yang digunakan
   - Grafik hasil
   - Timestamp
   - Pastikan username/nama terlihat (di browser title atau window)

## Format Screenshot:
- Format: JPG atau PNG
- Nama file: `X.monitoring_<nama_metric>.jpg`
- Resolusi: Minimal 1280x720
