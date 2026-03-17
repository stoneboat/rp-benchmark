#!/usr/bin/env bash
# Cluster installation for rp-benchmark on Purdue RCAC Bell
set -euo pipefail

echo "=== rp-benchmark cluster install (Bell) ==="

PROJ_DIR="$HOME/Desktop/rp-benchmark"
VENV_DIR="$PROJ_DIR/.venv"

module load anaconda/2024.02-py311 2>/dev/null || echo "anaconda module not found, using system python"

cd "$PROJ_DIR"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

python -c "import rpbench; print(f'rpbench {rpbench.__version__} imported OK')"

echo "=== Cluster install complete ==="
