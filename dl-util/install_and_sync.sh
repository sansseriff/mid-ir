#!/bin/sh
# Repo setup script: install uv if needed, sync env, and optionally start Jupyter Lab.
# This script lives in dl-util/ but operates from the repository root so relative paths work.

set -e

# Resolve repo root (parent of this script's directory)
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "[setup] Working directory: $REPO_ROOT"
echo "[setup] Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found. Installing via official script..."
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://astral.sh/uv/install.sh | sh || {
            echo "[error] Failed to install uv via curl. Please check your network or permissions.";
            exit 1;
        }
        # Make uv available immediately in this shell session
        export PATH="$HOME/.local/bin:$PATH"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh || {
            echo "[error] Failed to install uv via wget. Please check your network or permissions.";
            exit 1;
        }
        # Make uv available immediately in this shell session
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "[error] Neither curl nor wget is available to install uv. Please install one and re-run.";
        exit 1;
    fi
else
    echo "[setup] uv is already installed."
fi

echo "[setup] Syncing Python environment with uv..."
uv sync

# Optional: register a kernel name tied to this environment for convenience
if command -v uv >/dev/null 2>&1; then
    echo "[setup] Registering IPython kernel (optional)..."
    uv run ipython kernel install --user --name="project" || echo "[warn] ipython kernel registration skipped."
fi

# Prompt control: honor START_JUPYTER if provided, otherwise try prompting via /dev/tty
should_launch() {
    case "${START_JUPYTER:-}" in
        1|yes|YES|true|TRUE) return 0 ;;
        0|no|NO|false|FALSE) return 1 ;;
    esac

    if [ -e /dev/tty ] && [ -r /dev/tty ] && [ -w /dev/tty ]; then
        # Print a helpful hint in bright cyan, then prompt once
        printf "\033[1;36mNew to using uv to manage python projects? Check the \"uv Usage hints\" in the README of this project\033[0m\n" > /dev/tty
        printf "Start Jupyter Lab now? [Y/n]: " > /dev/tty
        IFS= read -r ans < /dev/tty || ans=""
        case "$ans" in
            [Nn]*|no|No|NO) return 1 ;;
            *) return 0 ;;
        esac
    fi
    return 1
}

if should_launch; then
    echo "[setup] Launching Jupyter Lab... (Ctrl+C to stop)"
    uv run --with jupyter jupyter lab
else
    echo "[setup] Skipping Jupyter Lab launch. You can start it later with: uv run --with jupyter jupyter lab"
fi
