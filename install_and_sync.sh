#!/bin/sh
# Script to install uv if not already installed, then run uv sync


if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing via official script..."
    wget -qO- https://astral.sh/uv/install.sh | sh || {
        echo "Failed to install uv. Please check your network or permissions.";
        exit 1;
    }
else
    echo "uv is already installed."
fi

uv sync

uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project

uv run --with jupyter jupyter lab
