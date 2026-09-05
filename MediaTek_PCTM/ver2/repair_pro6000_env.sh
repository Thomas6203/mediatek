#!/usr/bin/env bash
# Repair a broken Triton installation record, then install the RTX PRO 6000
# Blackwell-compatible project environment. Run once before run_dl.sh/run_rl.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m pip install --upgrade pip
# --ignore-installed does not try to uninstall the corrupt triton-3.0.0 entry
# that lacks RECORD/METADATA; it writes a fresh, valid Triton 3.3.1 record.
"$PYTHON_BIN" -m pip install --ignore-installed --no-deps --no-cache-dir triton==3.3.1
# The CUDA 12.8 PyTorch wheel required by RTX PRO 6000 Blackwell is pinned in
# requirements.txt. --ignore-installed avoids the same corrupt-record problem
# for any adjacent CUDA runtime package left by the old environment.
"$PYTHON_BIN" -m pip install --ignore-installed --no-cache-dir -r requirements.txt

"$PYTHON_BIN" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"CUDA runtime={torch.version.cuda}")
print(f"CUDA available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU={torch.cuda.get_device_name(0)}")
PY
