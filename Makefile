.PHONY: install lint format typecheck test train evaluate serve docker-build up down logs

install:
	uv sync --all-extras --dev
	uv pip install -e .

lint:
	uv run ruff check src scripts

format:
	uv run ruff format src scripts

typecheck:
	uv run mypy src

test:
	uv run pytest tests -q

train:
	uv run python scripts/train.py

evaluate:
	uv run python scripts/evaluate.py

serve:
	uv run python scripts/serve.py --host 0.0.0.0 --port 8000

docker-build:
	docker build -t mlsys-serve:latest -f docker/Dockerfile.serve .

up:
	docker compose -f docker/docker-compose.yml up --build

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f api
