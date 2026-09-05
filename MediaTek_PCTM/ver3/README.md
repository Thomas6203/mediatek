# ver3 — multi-agent Mamba RL Amazon suite

Independent snapshot for `run_amazons_full_rl.sh`, `run_mamba_rl.sh`, and the
multi-agent Mamba-RL modules.

Additional preserved material:

- `prepare_yelp19.py`: Yelp19 conversion utility
- `docs/latex/`: paper sources and generated PDF
- `archive/`: legacy tuning/backups and a full Git-history bundle
- `repair_pro6000_env.sh`: environment repair launcher

Run the seven Amazon subsets:

```bash
bash /workspace/P78123011/MediaTek_2026_project/ver3/run_amazons_full_rl.sh
```

Run one dataset directly:

```bash
bash /workspace/P78123011/MediaTek_2026_project/ver3/run_mamba_rl.sh amazon-all-beauty 0
```

Default paths:

- shared model/dataset cache: `/workspace/P78123011/cache/`
- results: `ver3/outputs_mamba_rl/`

Existing Mamba-RL results were migrated into `ver3/outputs_mamba_rl/`. The
legacy project-root `outputs_mamba_rl` path is a compatibility symlink to this
directory for runs that were already active during migration.

`run_amazons_full_rl.sh` defaults to GPU IDs `0,1`; override with, for example,
`GPU_IDS=0`. Set `CACHE_DIR` only when an alternative cache location is required.
