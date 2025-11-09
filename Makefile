.PHONY: install lint test train evaluate api docker-build up down logs inference

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

mypy:
	uv run mypy src

test:
	uv run pytest tests -q

train:
	uv run python scripts/train.py --models all

train-no-mlflow:
	uv run python scripts/train.py --models all --no-mlflow

evaluate:
	uv run python scripts/evaluate.py

api:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

inference: api

docker-build:
	docker build -t mlsys-api:latest .

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api
