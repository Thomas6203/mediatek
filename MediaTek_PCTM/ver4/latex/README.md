# ver4 paper

Main source: `preference_transition_mamba_rl.tex`

The report describes the implemented Preference-Transition Multi-LoRA Mamba
system and contains:

- the frozen Mamba-2.8B/LightGCN hybrid item representation;
- four functional agents with 13 disjoint, automatically routed LoRA adapters;
- latent preference prototype discovery;
- current-to-next preference transition prediction;
- four preference auxiliary losses and the final ranking equation;
- staged listwise ranking, future soft labels, hard-negative mining, and full-catalog evaluation;
- protocol-aware published MovieLens, Yelp, and Amazon result tables;
- the completed ver4 Amazon All Beauty multi-LoRA result available at generation time.

Run `bash build_latex.sh` with a full TeX Live installation. TikZ/PGF is
required for the two architecture figures.
