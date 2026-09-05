#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

xelatex -interaction=nonstopmode -halt-on-error multi_agent_mamba_rl.tex
xelatex -interaction=nonstopmode -halt-on-error multi_agent_mamba_rl.tex

if grep -Eq "Citation .* undefined|There were undefined references" multi_agent_mamba_rl.log; then
  echo "LaTeX build completed, but unresolved citations or references remain." >&2
  exit 1
fi

echo "Built multi_agent_mamba_rl.pdf with citations and cross-references resolved."
