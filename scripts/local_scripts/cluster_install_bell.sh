echo "Installing Environment for rp-benchmark (Bell)"

module load anaconda

# Initialize conda in this non-interactive shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

# Configure conda to handle SSL issues on cluster systems
echo "Updating conda certificates..."
conda update -n base -c defaults conda --yes 2>/dev/null || true

export REQUESTS_CA_BUNDLE=""
export CURL_CA_BUNDLE=""

# Set the conda environment path using the python-venv directory as prefix
mkdir -p /tmp/python-venv

CONDA_ENV_PATH="/tmp/python-venv/rp_benchmark_venv"

# Conda (recent versions) may require accepting Anaconda ToS for default channels.
# We try to accept non-interactively; if unavailable, we print the exact commands.
if conda tos --help >/dev/null 2>&1; then
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
else
    echo "Note: your conda may require accepting Anaconda channel ToS." >&2
    echo "If conda create fails with CondaToSNonInteractiveError, run:" >&2
    echo "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main" >&2
    echo "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r" >&2
fi

if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Conda env 'rp_benchmark_venv' already exists in $CONDA_ENV_PATH."
else
    echo "Creating conda env 'rp_benchmark_venv' in $CONDA_ENV_PATH..."
    # Retry logic for SSL issues
    MAX_RETRIES=3
    RETRY_COUNT=0
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if conda create --prefix "$CONDA_ENV_PATH" python=3.11 -y 2>&1; then
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                echo "Retry $RETRY_COUNT/$MAX_RETRIES: conda create failed, retrying in 5 seconds..."
                sleep 5
            else
                echo "conda create failed after $MAX_RETRIES attempts" >&2
                echo "Trying with SSL verification disabled..." >&2
                conda config --set ssl_verify false
                conda create --prefix "$CONDA_ENV_PATH" python=3.11 -y || { echo "conda create failed even with SSL disabled" >&2; exit 1; }
            fi
        fi
    done
fi

# Activate the conda environment
conda activate "$CONDA_ENV_PATH"

# Upgrade pip
pip install --upgrade pip

# Install packages from requirements.txt
echo "Installing pip packages..."
pip install -r requirements.txt || { echo "pip install failed" >&2; exit 1; }

# Install + register the kernel (requires Jupyter to be available on the system).
echo "Installing ipykernel..."
pip install -q ipykernel

KERNEL_NAME="rpbench"
KERNEL_DISPLAY_NAME="Python (rpbench)"
python -m ipykernel install --user --name="${KERNEL_NAME}" --display-name "${KERNEL_DISPLAY_NAME}" || { echo "kernel registration failed" >&2; exit 1; }

# Deactivate conda environment
conda deactivate

echo "=== Cluster install complete ==="
echo "Conda env: $CONDA_ENV_PATH"
echo "Kernel: ${KERNEL_DISPLAY_NAME} (name: ${KERNEL_NAME})"