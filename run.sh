#!/bin/bash
# Helper script to run commands with uv

set -e

# Deactivate any active virtual env
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current virtual environment: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
fi

# Run command with uv
echo "Running with uv: $@"
uv run "$@"
