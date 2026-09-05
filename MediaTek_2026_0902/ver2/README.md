# ver2 — REINFORCE RL

Independent snapshot for `run_rl.sh` and `src.train_rl`.

Environment repair is available through `bash repair_pro6000_env.sh`.

Run from any directory:

```bash
bash /workspace/P78123011/MediaTek_2026_project/ver2/run_rl.sh 0
```

Default paths:

- shared model/dataset cache and LoRA adapter: `/workspace/P78123011/cache/`
- logs: `ver2/log/`
- figures: `ver2/fig_outputs/`
- JSON samples: `ver2/json_outputs/`

Existing `rl_*.log` files were migrated into `ver2/log/`.

Set `CACHE_DIR` only when an alternative cache location is required.
