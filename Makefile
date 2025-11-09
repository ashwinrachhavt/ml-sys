.PHONY: install lint format typecheck test train evaluate serve docker-build up down logs

install:
	python -m pip install --upgrade pip
	pip install .[dev]

lint:
	ruff check src scripts

format:
	ruff format src scripts

typecheck:
	mypy src

test:
	pytest tests -q

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

serve:
	python scripts/serve.py --host 0.0.0.0 --port 8000

docker-build:
	docker build -t mlsys-serve:latest -f docker/Dockerfile.serve .

up:
	docker compose -f docker/docker-compose.yml up --build

down:
	docker compose -f docker/docker-compose.yml down

logs:
	docker compose -f docker/docker-compose.yml logs -f api
