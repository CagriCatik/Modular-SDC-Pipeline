#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "Setting up Modular-SDC-Pipeline..."

# 1. Install system dependencies (swig is required to build box2d-py for CarRacing-v3)
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        if [[ -x /opt/homebrew/bin/brew ]]; then
            export PATH="/opt/homebrew/bin:$PATH"
        elif [[ -x /usr/local/bin/brew ]]; then
            export PATH="/usr/local/bin:$PATH"
        else
            echo "Error: Homebrew is required on macOS. Install it from https://brew.sh" >&2
            exit 1
        fi
    fi

    if ! command -v swig >/dev/null 2>&1; then
        echo "Installing swig via Homebrew..."
        brew install swig
    fi
elif [[ "$OS" == "Linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing system dependencies via apt-get..."
        sudo apt-get update
        sudo apt-get install -y python3-dev python3-pip python3-venv swig libgl1-mesa-glx libglu1-mesa
    else
        echo "Warning: apt-get not found. Ensure python3 (>=3.10), venv, and swig are installed."
    fi
fi

# 2. Locate Python 3.10+
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    if [[ "$OS" == "Darwin" ]]; then
        echo "Python 3.10+ not found. Installing python@3.11 via Homebrew..."
        brew install python@3.11
        PYTHON_BIN="$(brew --prefix python@3.11)/bin/python3.11"
    else
        echo "Error: Python 3.10+ is required." >&2
        exit 1
    fi
fi

echo "Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

# 3. Create virtual environment and install dependencies
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment in .venv..."
    "$PYTHON_BIN" -m venv .venv
fi

echo "Installing Python dependencies (including Gymnasium Box2D / CarRacing-v3)..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt pytest

# 4. Verify CarRacing-v3 environment
echo "Verifying CarRacing-v3 environment..."
.venv/bin/python -c 'import gymnasium as gym; env = gym.make("CarRacing-v3", render_mode="rgb_array"); env.reset(seed=0); env.close(); print("CarRacing-v3 ready!")'

echo ""
echo "Setup complete! To run the pipeline:"
echo "  source .venv/bin/activate"
echo "  python modular_pipeline.py"
