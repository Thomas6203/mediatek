#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

legacy_pattern='/workspace/P78123011/MediaTek_2026_project/(run_mamba_rl\.sh|run_amazons_full_rl\.sh)|-m [s]rc\.train'
if pgrep -af "$legacy_pattern" >/dev/null; then
  echo "NOT SAFE: a process launched from the legacy project root is still running:" >&2
  pgrep -af "$legacy_pattern" >&2
  exit 1
fi

required=(
  "$PROJECT_ROOT/README.md"
  "$PROJECT_ROOT/ver1/run.sh"
  "$PROJECT_ROOT/ver1/src/train.py"
  "$PROJECT_ROOT/ver2/run_rl.sh"
  "$PROJECT_ROOT/ver2/src/train_rl.py"
  "$PROJECT_ROOT/ver3/run_amazons_full_rl.sh"
  "$PROJECT_ROOT/ver3/run_mamba_rl.sh"
  "$PROJECT_ROOT/ver3/src/train_mamba_rl.py"
  "$PROJECT_ROOT/ver3/outputs_mamba_rl"
  "$PROJECT_ROOT/ver3/docs/latex/multi_agent_mamba_rl.tex"
  "$PROJECT_ROOT/ver3/archive/MediaTek_2026_project.bundle"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "NOT SAFE: required preserved path is missing: $path" >&2
    exit 1
  fi
done

git bundle verify "$PROJECT_ROOT/ver3/archive/MediaTek_2026_project.bundle" >/dev/null
bash -n \
  "$PROJECT_ROOT/ver1/run.sh" \
  "$PROJECT_ROOT/ver2/run_rl.sh" \
  "$PROJECT_ROOT/ver3/run_mamba_rl.sh" \
  "$PROJECT_ROOT/ver3/run_amazons_full_rl.sh"

echo "SAFE: no legacy training process is running and all preserved paths passed verification."
