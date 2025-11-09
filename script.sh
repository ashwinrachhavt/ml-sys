#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEMP_DIR=$(mktemp -d)
CONFIG_FILE="$TEMP_DIR/e2e-config.yaml"
PORT=${PORT:-8001}

trap 'rm -rf "$TEMP_DIR"; if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" >/dev/null 2>&1 || true; fi' EXIT

cat >"$CONFIG_FILE" <<YAML
project:
  name: "e2e"
  version: "0.0.1"

data:
  loader_type: "csv"
  sources:
    customers: "${ROOT_DIR}/dataset/raw/customers.csv"
    noncustomers: "${ROOT_DIR}/dataset/raw/noncustomers.csv"
    usage_actions: "${ROOT_DIR}/dataset/raw/usage_actions.csv"
  id_column: "id"
  target_column: "is_customer"

features:
  transformers:
    - type: "datetime"
      columns: ["CLOSEDATE"]
      reference_date: "2024-01-01"
    - type: "categorical"
      columns: ["EMPLOYEE_RANGE", "INDUSTRY"]
      encoding: "onehot"
    - type: "fillna"

training:
  models:
    - name: "logistic_regression"
      type: "sklearn.logistic_regression"
      param_grid:
        C: [0.1, 1.0]
        max_iter: [200]
  cv_folds: 2
  test_size: 0.2
  val_size: 0.2
  random_state: 42
  stratify: false

evaluation:
  primary_metric: "roc_auc"
  threshold_metric: "f1"
  metrics: ["roc_auc", "f1", "accuracy"]

tracking:
  backend: "none"

serving:
  model_name: "e2e-model"
  model_stage: "Staging"
  host: "0.0.0.0"
  port: ${PORT}
  local_model_path: "$TEMP_DIR/model.joblib"
YAML

Step() {
  echo "[E2E] $1" >&2
}

Step "Training pipeline"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
python "$ROOT_DIR/scripts/train.py" --config "$CONFIG_FILE" --no-tracking

Step "Checking health endpoint via TestClient"
python - <<PY
import json
from fastapi.testclient import TestClient

import os

from mlsys.api import create_app
from mlsys.config import Settings

settings = Settings.from_yaml("$CONFIG_FILE")
client = TestClient(create_app(settings=settings))

response = client.get("/api/health")
response.raise_for_status()
print(json.dumps(response.json()))
PY

Step "Calling predict endpoint"
export E2E_CONFIG="$CONFIG_FILE"
if python - <<'PY'
import json
import os
from fastapi.testclient import TestClient

from mlsys.api import create_app
from mlsys.config import Settings

sample = {
    "payload": {
        "id": 1,
        "CLOSEDATE": "2020-01-01",
        "EMPLOYEE_RANGE": "51 to 200",
        "INDUSTRY": "SOFTWARE",
    }
}

settings = Settings.from_yaml(os.environ["E2E_CONFIG"])
client = TestClient(create_app(settings=settings))

response = client.post("/api/predict", json=sample, timeout=10)
print("status", response.status_code)
print(response.text)
response.raise_for_status()
PY
then
    predict_status=0
else
    predict_status=$?
fi

if [[ "${predict_status:-0}" -ne 0 ]]; then
  echo "[E2E] Predict endpoint failed" >&2
  exit "${predict_status}"
fi

Step "E2E completed"
