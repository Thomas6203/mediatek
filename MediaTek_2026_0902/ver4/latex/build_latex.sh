#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

if command -v xelatex >/dev/null 2>&1; then
  LATEX_ENGINE="xelatex"
elif command -v pdflatex >/dev/null 2>&1; then
  LATEX_ENGINE="pdflatex"
elif command -v lualatex >/dev/null 2>&1; then
  LATEX_ENGINE="lualatex"
else
  echo "xelatex, pdflatex, or lualatex is required." >&2
  exit 127
fi

"$LATEX_ENGINE" -interaction=nonstopmode -halt-on-error preference_transition_mamba_rl.tex
"$LATEX_ENGINE" -interaction=nonstopmode -halt-on-error preference_transition_mamba_rl.tex

if grep -Eq "Citation .* undefined|There were undefined references" preference_transition_mamba_rl.log; then
  echo "LaTeX build completed, but unresolved citations or references remain." >&2
  exit 1
fi

echo "Built preference_transition_mamba_rl.pdf with citations and cross-references resolved."
