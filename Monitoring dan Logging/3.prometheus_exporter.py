"""
Prometheus Metrics Exporter for Ethereum Fraud Detection Model
Author: Mohammad Ari Alexander Aziz
Description: Custom metrics exporter for model monitoring

This script exposes custom metrics for Prometheus to scrape.
Run this alongside your inference service.
"""

from prometheus_client import start_http_server, Counter, Histogram, Gauge, Summary
import time
import random

# Define custom metrics

# 1. Request Counters
total_predictions = Counter(
    'fraud_detection_predictions_total',
    'Total number of fraud detection predictions made',
    ['model_version', 'prediction_result']
)

prediction_errors = Counter(
    'fraud_detection_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# 2. Prediction Distribution
prediction_confidence = Histogram(
    'fraud_detection_confidence',
    'Distribution of prediction confidence scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

# 3. Response Time
prediction_latency = Histogram(
    'fraud_detection_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

# Alternative: Summary for percentiles
prediction_latency_summary = Summary(
    'fraud_detection_latency_summary',
    'Summary of prediction latency'
)

# 4. System Metrics
active_connections = Gauge(
    'fraud_detection_active_connections',
    'Number of active connections to the prediction service'
)

model_memory_usage = Gauge(
    'fraud_detection_model_memory_mb',
    'Memory usage of the model in MB'
)

cpu_usage = Gauge(
    'fraud_detection_cpu_usage_percent',
    'CPU usage percentage of the prediction service'
)

# 5. Model Performance Metrics (update these periodically)
model_accuracy = Gauge(
    'fraud_detection_model_accuracy',
    'Current model accuracy on validation set'
)

model_precision = Gauge(
    'fraud_detection_model_precision',
    'Current model precision'
)

model_recall = Gauge(
    'fraud_detection_model_recall',
    'Current model recall'
)

model_f1_score = Gauge(
    'fraud_detection_model_f1_score',
    'Current model F1 score'
)

# 6. Data Drift Metrics
feature_drift_score = Gauge(
    'fraud_detection_feature_drift',
    'Feature drift score compared to training data',
    ['feature_name']
)

# 7. Business Metrics
fraud_rate = Gauge(
    'fraud_detection_fraud_rate',
    'Percentage of transactions predicted as fraud'
)

daily_transactions = Counter(
    'fraud_detection_daily_transactions_total',
    'Total transactions processed daily'
)


class MetricsExporter:
    """Class to manage and export metrics"""

    def __init__(self, model_version='1.0.0'):
        self.model_version = model_version
        self.fraud_count = 0
        self.total_count = 0

        # Initialize model performance metrics (from training)
        model_accuracy.set(0.9756)  # Example: Set to your actual model accuracy
        model_precision.set(0.9612)
        model_recall.set(0.9541)
        model_f1_score.set(0.9576)

    def record_prediction(self, prediction, confidence, latency):
        """
        Record a prediction with its metrics

        Args:
            prediction: 0 (non-fraud) or 1 (fraud)
            confidence: confidence score (0-1)
            latency: processing time in seconds
        """
        # Update counters
        result = 'fraud' if prediction == 1 else 'non_fraud'
        total_predictions.labels(
            model_version=self.model_version,
            prediction_result=result
        ).inc()

        daily_transactions.inc()

        # Update distribution metrics
        prediction_confidence.observe(confidence)
        prediction_latency.observe(latency)
        prediction_latency_summary.observe(latency)

        # Update fraud rate
        self.total_count += 1
        if prediction == 1:
            self.fraud_count += 1
        fraud_rate.set((self.fraud_count / self.total_count) * 100)

    def record_error(self, error_type='unknown'):
        """Record a prediction error"""
        prediction_errors.labels(error_type=error_type).inc()

    def update_system_metrics(self, connections, memory_mb, cpu_percent):
        """Update system resource metrics"""
        active_connections.set(connections)
        model_memory_usage.set(memory_mb)
        cpu_usage.set(cpu_percent)

    def update_drift_metrics(self, feature_drifts):
        """
        Update feature drift metrics

        Args:
            feature_drifts: dict of {feature_name: drift_score}
        """
        for feature_name, drift_score in feature_drifts.items():
            feature_drift_score.labels(feature_name=feature_name).set(drift_score)


def simulate_metrics():
    """Simulate metrics for demonstration (remove in production)"""
    exporter = MetricsExporter(model_version='1.0.0')

    print("Metrics exporter started. Simulating predictions...")
    print("Access metrics at: http://localhost:8000/metrics")

    while True:
        # Simulate prediction
        prediction = random.choices([0, 1], weights=[0.8, 0.2])[0]  # 20% fraud rate
        confidence = random.uniform(0.6, 0.99)
        latency = random.uniform(0.001, 0.1)

        exporter.record_prediction(prediction, confidence, latency)

        # Simulate system metrics
        connections = random.randint(1, 10)
        memory = random.uniform(50, 150)
        cpu = random.uniform(5, 40)

        exporter.update_system_metrics(connections, memory, cpu)

        # Simulate occasional errors
        if random.random() < 0.01:  # 1% error rate
            error_type = random.choice(['timeout', 'invalid_input', 'model_error'])
            exporter.record_error(error_type)

        # Simulate feature drift (update every 100 iterations)
        if exporter.total_count % 100 == 0:
            drift_metrics = {
                'avg_transaction_value': random.uniform(0, 0.3),
                'transaction_frequency': random.uniform(0, 0.2),
                'unique_addresses': random.uniform(0, 0.15)
            }
            exporter.update_drift_metrics(drift_metrics)

        time.sleep(1)  # Generate metrics every second


if __name__ == '__main__':
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    print("="*70)
    print("Prometheus Metrics Exporter Started")
    print("="*70)
    print("Metrics available at: http://localhost:8000/metrics")
    print("Prometheus should scrape this endpoint")
    print("="*70)

    # Simulate metrics (in production, integrate with your actual inference service)
    try:
        simulate_metrics()
    except KeyboardInterrupt:
        print("\nMetrics exporter stopped.")
