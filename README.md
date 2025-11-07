# ml-sys

## Environment setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev --extra ui
```

All commands below assume the virtual environment created by `uv sync`; prefix them with `uv run` to execute inside it.

## Running the system

### Retraining

```bash
uv run python -m mlsys.training.cli --tune
```

Artifacts land in `artifacts/`.

### Inference API

```bash
uv run uvicorn mlsys.inference.service:app --factory --reload
```

- `/health` returns `{ "status": "ok" }`
- `/score` accepts `{ "leads": [ ... ] }`

### Streamlit Evaluator

```bash
uv run streamlit run src/mlsys/ui/app.py
```

Set the API URL (defaults to `http://localhost:8000`) and either upload a CSV or load a sample from the sidebar.
