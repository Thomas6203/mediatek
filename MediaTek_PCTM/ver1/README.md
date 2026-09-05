# ver1 — supervised DL

Independent snapshot for `run.sh` and `src.train`.

`run_dl.sh` is also retained as the legacy equivalent launcher. Environment
repair is available through `bash repair_pro6000_env.sh`.

Run from any directory:

```bash
bash /workspace/P78123011/MediaTek_2026_project/ver1/run.sh 0
```

Default paths:

- shared model/dataset cache: `/workspace/P78123011/cache/`
- logs: `ver1/log/`
- figures: `ver1/fig_outputs/`
- JSON samples: `ver1/json_outputs/`

Existing supervised-DL logs, loss figures, and sample JSON files were migrated
into these directories.

Set `CACHE_DIR` only when an alternative cache location is required.
