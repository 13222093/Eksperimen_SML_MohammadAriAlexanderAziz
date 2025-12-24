"""
Inference API for Ethereum Fraud Detection Model
Author: Mohammad Ari Alexander Aziz
Description: REST API for model serving with Prometheus metrics integration
"""

from flask import Flask, request, jsonify
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import pickle
import numpy as np
import pandas as pd
import time
import psutil
import os

app = Flask(__name__)

# Add Prometheus metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# Prometheus metrics
prediction_counter = Counter(
    'fraud_predictions_total',
    'Total predictions made',
    ['prediction']
)

prediction_latency = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency'
)

fraud_rate_gauge = Gauge(
    'fraud_rate_current',
    'Current fraud detection rate'
)

active_requests_gauge = Gauge(
    'active_requests',
    'Number of active prediction requests'
)

# Global variables
model = None
scaler = None
feature_names = None
fraud_count = 0
total_count = 0


def load_model_and_scaler(model_path='best_xgboost.pkl', scaler_path='scaler_model.pkl'):
    """Load the trained model and scaler"""
    global model, scaler

    print("Loading model and scaler...")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {scaler_path}")

        return True
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Make sure model and scaler files are in the same directory")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'total_predictions': total_count,
        'fraud_detections': fraud_count
    }
    return jsonify(status), 200 if model is not None else 503


@app.route('/predict', methods=['POST'])
@prediction_latency.time()
def predict():
    """Prediction endpoint"""
    global fraud_count, total_count

    active_requests_gauge.inc()

    try:
        start_time = time.time()

        # Get input data
        data = request.get_json()

        if 'features' not in data:
            active_requests_gauge.dec()
            return jsonify({'error': 'Missing features in request'}), 400

        # Convert to numpy array
        features = np.array(data['features']).reshape(1, -1)

        # Check if we need to scale
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Update metrics
        total_count += 1
        if prediction == 1:
            fraud_count += 1
            prediction_counter.labels(prediction='fraud').inc()
        else:
            prediction_counter.labels(prediction='non_fraud').inc()

        # Update fraud rate
        fraud_rate = (fraud_count / total_count) * 100
        fraud_rate_gauge.set(fraud_rate)

        # Calculate latency
        latency = time.time() - start_time

        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Fraud' if prediction == 1 else 'Non-Fraud',
            'confidence': {
                'non_fraud': float(prediction_proba[0]),
                'fraud': float(prediction_proba[1])
            },
            'latency_ms': round(latency * 1000, 2),
            'model_version': '1.0.0'
        }

        active_requests_gauge.dec()
        return jsonify(response), 200

    except Exception as e:
        active_requests_gauge.dec()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    global fraud_count, total_count

    try:
        data = request.get_json()

        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400

        # Convert to numpy array
        features = np.array(data['features'])

        # Scale if needed
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # Make predictions
        predictions = model.predict(features_scaled)
        predictions_proba = model.predict_proba(features_scaled)

        # Update metrics
        for pred in predictions:
            total_count += 1
            if pred == 1:
                fraud_count += 1
                prediction_counter.labels(prediction='fraud').inc()
            else:
                prediction_counter.labels(prediction='non_fraud').inc()

        # Update fraud rate
        fraud_rate = (fraud_count / total_count) * 100
        fraud_rate_gauge.set(fraud_rate)

        # Prepare response
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            results.append({
                'index': i,
                'prediction': int(pred),
                'prediction_label': 'Fraud' if pred == 1 else 'Non-Fraud',
                'confidence': {
                    'non_fraud': float(proba[0]),
                    'fraud': float(proba[1])
                }
            })

        response = {
            'predictions': results,
            'total_processed': len(predictions),
            'fraud_detected': int(np.sum(predictions)),
            'model_version': '1.0.0'
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics"""
    stats = {
        'total_predictions': total_count,
        'fraud_predictions': fraud_count,
        'non_fraud_predictions': total_count - fraud_count,
        'fraud_rate_percent': round((fraud_count / total_count * 100) if total_count > 0 else 0, 2),
        'system': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'process_memory_mb': round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
        }
    }
    return jsonify(stats), 200


@app.route('/', methods=['GET'])
def index():
    """API information endpoint"""
    info = {
        'service': 'Ethereum Fraud Detection API',
        'version': '1.0.0',
        'author': 'Mohammad Ari Alexander Aziz',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Single prediction (POST)',
            '/batch_predict': 'Batch predictions (POST)',
            '/stats': 'Prediction statistics (GET)',
            '/metrics': 'Prometheus metrics (GET)'
        },
        'status': 'running'
    }
    return jsonify(info), 200


if __name__ == '__main__':
    print("="*70)
    print("Ethereum Fraud Detection - Inference API")
    print("="*70)

    # Load model and scaler
    model_loaded = load_model_and_scaler()

    if not model_loaded:
        print("\nWARNING: Model or scaler not loaded!")
        print("Place 'best_xgboost.pkl' and 'scaler_model.pkl' in this directory")
        print("Continuing anyway for demonstration...")

    print("\nStarting Flask server...")
    print("API endpoints:")
    print("  - http://localhost:5001/")
    print("  - http://localhost:5001/health")
    print("  - http://localhost:5001/predict")
    print("  - http://localhost:5001/batch_predict")
    print("  - http://localhost:5001/stats")
    print("  - http://localhost:5001/metrics (Prometheus)")
    print("="*70)

    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
