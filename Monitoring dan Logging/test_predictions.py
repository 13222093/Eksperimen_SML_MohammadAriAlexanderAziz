"""
Test Script to Generate Predictions
This generates traffic to populate Prometheus/Grafana metrics
"""

import requests
import random
import time

# API endpoint
url = "http://localhost:5001/predict"

print("="*70)
print("Generating Test Predictions for Prometheus/Grafana")
print("="*70)
print(f"\nTargeting: {url}")
print("Generating 100 predictions...\n")

success_count = 0
error_count = 0

for i in range(100):
    # Generate random feature values (16 features for the model)
    features = [random.uniform(0, 1) for _ in range(16)]

    try:
        response = requests.post(url, json={"features": features})

        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction_label', 'Unknown')
            confidence = result.get('confidence', {})

            print(f"[{i+1}/100] Prediction: {prediction} | "
                  f"Fraud confidence: {confidence.get('fraud', 0):.2%}")
            success_count += 1
        else:
            print(f"[{i+1}/100] Error: HTTP {response.status_code}")
            error_count += 1

    except requests.exceptions.ConnectionError:
        print(f"[{i+1}/100] Error: Could not connect to API. Is 7.Inference.py running?")
        error_count += 1
        break
    except Exception as e:
        print(f"[{i+1}/100] Error: {e}")
        error_count += 1

    # Wait between requests
    time.sleep(0.5)

print("\n" + "="*70)
print("Test Complete!")
print("="*70)
print(f"Successful predictions: {success_count}")
print(f"Errors: {error_count}")
print("\nNow check:")
print("- Prometheus UI: http://localhost:9090")
print("- Grafana UI: http://localhost:3000")
print("="*70)
