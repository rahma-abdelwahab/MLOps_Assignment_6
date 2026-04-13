import os
import mlflow

THRESHOLD = 0.85

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking accuracy for Run ID: {run_id}")

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("ERROR: No accuracy metric found in MLflow run.")
    exit(1)

print(f"Accuracy: {accuracy:.4f} | Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: Accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    exit(1)

print(f"PASSED: Accuracy {accuracy:.4f} meets threshold. Proceeding to deploy.")