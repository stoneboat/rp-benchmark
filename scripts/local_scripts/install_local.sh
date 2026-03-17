#!/usr/bin/env bash
# Local installation script for rp-benchmark
set -euo pipefail

echo "=== rp-benchmark local install ==="

# Create venv if needed
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Smoke import check
python -c "import rpbench; print(f'rpbench {rpbench.__version__} imported OK')"

echo "=== Install complete ==="
