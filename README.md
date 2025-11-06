# ml-sys

## Running the system

### Retraining

```bash
PYTHONPATH=src /Users/ashwin/Applications/Master/venv/bin/python -m mlsys.training.cli --tune
```

Artifacts land in `artifacts/`.

### Inference API

```bash
uvicorn mlsys.inference.service:app --factory --reload
```

- `/health` returns `{ "status": "ok" }`
- `/score` accepts `{ "leads": [ ... ] }`

### Streamlit Evaluator

```bash
streamlit run src/mlsys/ui/app.py
```

Set the API URL (defaults to `http://localhost:8000`) and either upload a CSV or load a sample from the sidebar.
