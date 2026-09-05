# ver4: Preference-Transition Multi-Agent Mamba Ranker

ver4 is an independent copy of the ver3 training system. It does not import
code, logs, or outputs from ver1, ver2, or ver3. Shared downloaded datasets and
pretrained-model files remain in `/workspace/P78123011/cache`.

## Run

```bash
cd /workspace/P78123011/MediaTek_2026_project/ver4
bash run_amazons_full_rl.sh
```

The subset selection remains exactly as it was in ver3 at the time ver4 was
created. Uncomment or comment the `run_subset` lines in
`run_amazons_full_rl.sh` as needed. Outputs are written only under
`ver4/outputs_mamba_rl`.

## Preference agent

Each item representation is softly assigned to one of a shared set of learned
preference prototypes. A GRU reads the preference sequence and produces:

- the current preference distribution;
- the predicted next-preference distribution;
- a probability that the next interaction represents a preference transition;
- preference-based candidate scores fused into the coordinator ranking.

The improved encoder also runs a tiny selective-state Mamba branch in parallel
with the GRU. Its input-dependent decay preserves or forgets latent preference
evidence in linear time. A near-zero residual gate (about 0.01 initially)
protects the GRU baseline while allowing training to increase the branch's
contribution. The coordinator makes the preference ranking weight
user-dependent using transition probability and prediction confidence.

Frozen Mamba item encoding uses a short preference-aware representation
prefix. Prompt tokens are excluded from mean pooling, and the prompt
participates in the cache fingerprint so prompted vectors cannot accidentally
reuse the older cache.

The preference prototypes and transition model are shared unchanged across all
Amazon subsets. Four auxiliary objectives train next-preference prediction,
transition detection, balanced prototype use, and prototype separation.

Every completed run writes the existing ranking artifacts plus
`preference_analysis.json`. The latter contains aggregate transition/entropy
statistics and per-user examples with the top current and predicted preference
IDs. These preference IDs are latent components, not human-authored labels.

No model weights are written by the Amazon suite because
`SAVE_MODEL_WEIGHTS=0`; logs, score JSON, recommendations,
`preference_analysis.json`, metrics, and `loss.png` are retained.

## Main preference hyperparameters

All can be set as environment variables before launching:

- `PREFERENCE_COUNT` (default 64)
- `PREFERENCE_HIDDEN` (default 128)
- `PREFERENCE_TEMPERATURE` (default 0.2)
- `PREFERENCE_SCORE_WEIGHT` (default 0.2)
- `PREFERENCE_TINY_MAMBA_DIM` (default 32)
- `PREFERENCE_COEF` (default 0.2)
- `PREFERENCE_TRANSITION_COEF` (default 0.1)
- `PREFERENCE_BALANCE_COEF` (default 0.01)
- `PREFERENCE_SEPARATION_COEF` (default 0.01)
- `ITEM_PROMPT_PREFIX` (default preference-aware product representation)

## Verification

```bash
python -m unittest discover -s tests
bash -n run_mamba_rl.sh
bash -n run_amazons_full_rl.sh
```

## LaTeX report

The ver4 method and protocol-aware literature benchmark report is located at
`latex/preference_transition_mamba_rl.tex`. From the `latex` directory, run
`bash build_latex.sh` to compile it.

The report requires a standard TeX Live installation with XeLaTeX or
pdfLaTeX, TikZ/PGF, AMSMath, booktabs, longtable, tabularx, listings,
hyperref, microtype, and fancyhdr. The minimal TeX installation currently
available on this machine does not include TikZ/PGF.
