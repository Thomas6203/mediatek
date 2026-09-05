# Independent runnable versions

| Folder | Entry point | Implementation |
|---|---|---|
| `ver1/` | `bash ver1/run.sh` | supervised DL recommender |
| `ver2/` | `bash ver2/run_rl.sh` | REINFORCE + LoRA Mamba |
| `ver3/` | `bash ver3/run_amazons_full_rl.sh` | multi-agent Mamba-RL Amazon suite |

Each folder contains its own `src` package, requirements file, and output roots.
All versions intentionally share `/workspace/P78123011/cache` for model and
dataset artifacts. Editing one version does not change the source imported by
the other versions. GPU devices are external shared resources, so choose
distinct CUDA IDs if versions are launched concurrently.

Historical outputs were classified and moved into their matching version. The
legacy `outputs_mamba_rl` name points to `ver3/outputs_mamba_rl` only for
backward compatibility with an older run that was active during migration.
