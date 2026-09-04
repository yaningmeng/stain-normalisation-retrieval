#!/usr/bin/env bash
# Create/update the environment used by the PLISM preprocessing and analysis scripts.

set -euo pipefail

ENV_NAME="plism"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH."
    echo "Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# conda activate is a shell function, so initialise it before activation.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Environment '$ENV_NAME' exists; updating it."
    conda env update --name "$ENV_NAME" --file "$REPO_DIR/environment.yml" --prune
else
    echo "==> Creating environment '$ENV_NAME'."
    conda env create --file "$REPO_DIR/environment.yml"
fi

echo "==> Verifying the installation."
conda activate "$ENV_NAME"

python - <<'PY'
import importlib
import shutil
import subprocess
import sys

REQUIRED = [
    "torch", "torchvision", "transformers", "huggingface_hub",
    "pyarrow", "pandas", "h5py", "PIL",
    "numpy", "scipy", "skimage", "sklearn", "matplotlib", "umap",
    "torch_staintools", "kornia", "tqdm",
]

missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{name}: {exc}")

if missing:
    print("FAILED to import:")
    for module in missing:
        print("   -", module)
    sys.exit(1)

import torch

print(f"python           {sys.version.split()[0]}")
print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu              {torch.cuda.get_device_name(0)}")
else:
    has_gpu = shutil.which("nvidia-smi") is not None and (
        subprocess.run(["nvidia-smi"], capture_output=True, check=False).returncode == 0
    )
    if has_gpu:
        print()
        print("WARNING: this machine has an NVIDIA GPU but torch cannot use it.")
        print("Check the CUDA version with nvidia-smi and install a matching PyTorch build.")

print("All imports OK.")
PY

cat <<EOF

==> Done. Activate the environment with:

    conda activate $ENV_NAME

Then, from the repository root, run for example:

    python scripts/plism_loader.py fetch --tiles-per-slide 400 --out-dir tiles
    python scripts/extract_features.py --tiles-dir tiles --model phikon_v2 \
        --normalise none --out-dir features/phikon_v2_none

EOF
