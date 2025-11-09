#!/usr/bin/env python3
"""Launch the FastAPI application."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from mlsys.api import create_app
from mlsys.config import Settings

DEFAULT_CONFIG = Path("config/config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the FastAPI application")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_yaml(args.config)
    app = create_app(settings=settings)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
