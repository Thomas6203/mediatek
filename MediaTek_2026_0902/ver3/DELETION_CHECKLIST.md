# Root cleanup safety

The project can be retained using only `ver1/`, `ver2/`, `ver3/`, and the
root `README.md` after all older training processes have stopped.

Preserved before cleanup:

- runnable source snapshots and launchers for all three versions;
- historical DL, RL, and multi-agent output files;
- paper sources, generated PDF, Yelp preparation, and legacy tuning tools;
- original `.orig` backups and synthetic cache;
- committed Git history in `archive/MediaTek_2026_project.bundle`.

Do not delete the root compatibility link or legacy launchers while a process
started from the old project root is still running. Check with:

```bash
bash ver3/check_before_root_cleanup.sh
```

Only proceed when this prints `SAFE` and exits successfully.

After cleanup, launch the Amazon suite only with:

```bash
cd /workspace/P78123011/MediaTek_2026_project/ver3
bash run_amazons_full_rl.sh
```

The Git repository can later be recovered from the bundle with:

```bash
git clone ver3/archive/MediaTek_2026_project.bundle recovered-project
```
