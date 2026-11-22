# Launching runs

Every training run starts the same way: through `tools/launch_experiment.sh`, which reads the
experiment's config from `experiments/experiments.json`, runs the preflight checks, and starts
`python -m gogeta.train --config <cfg> --compile --require-clean` **detached**, logging to Trackio.
This folder has one small script per experiment (`E<n>.sh`) plus `status.sh` and `stop.sh`.

**Rule: every new experiment gets a launch script here.** When you register `E<n>` in
`experiments/experiments.json`, add `experiments/launch/E<n>.sh` too. Until it has a committed
config *and* prediction, the script refuses with "not designed yet" (see `E4.sh`-`E8.sh`). Once
both exist, its body is one line (copy `E3.sh`):

```bash
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E<n> "$@"
```

All commands below run from the repo root. In WSL that is `/mnt/c/Users/mreca/Desktop/Gogeta-0`;
from Windows, use the `.ps1` wrappers in `tools/` (they call the `.sh` through `wsl -d Ubuntu-24.04`).

## Compute budget (AGENTS.md section 5)

To keep the laptop from overheating: **at most 1 GPU job at a time** (any size) and **at most 8
smaller CPU-only jobs** alongside it. The launchers enforce this:

| What | How it is enforced |
|---|---|
| 1 GPU job | `launch_experiment.sh` takes the single GPU slot, `~/fusiondance-data/run/gpu.lock` (pid, experiment, seed, log). A wrapper process holds it and removes it when training exits (normally, on error, or on SIGTERM/Ctrl-C). A lock whose PID is dead is stale and is removed by the next launch. Launches also refuse while `nvidia-smi` shows any GPU compute process that did not come through the launcher (e.g. a check script started by hand). |
| GPU temperature | Launches refuse at **>= 85 C**. Wait until it is **< 75 C**, or add `--wait-cool` to poll every 30 s until then. |
| 8 CPU jobs | `tools/launch_cpu_job.sh` refuses the 9th live job. Each job gets `CUDA_VISIBLE_DEVICES=""` (no GPU) and `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` (plus OpenBLAS/NumExpr = 2). |

GPU work that is not a registered experiment (benchmarks, GPU checks) is not covered by a launcher
yet: run it only when `status.sh` shows the slot free and no GPU process.

## 1. Start the dashboard (Trackio, local only)

```bash
bash tools/launch_tracker.sh start            # idempotent: if 7861 already serves it, just prints the URL
bash tools/launch_tracker.sh status           # UP/DOWN, pid, whether this script started it
bash tools/launch_tracker.sh url              # http://127.0.0.1:7861
bash tools/launch_tracker.sh stop             # stops only a dashboard this script started (PID file)
bash tools/launch_tracker.sh start --port 7862 --project gogeta-0-smoke
```

```powershell
.\tools\launch_tracker.ps1 start -Open        # starts if needed, then opens the browser
.\tools\launch_tracker.ps1 status
```

It binds 127.0.0.1 with sharing disabled. PID file `~/fusiondance-data/run/tracker-<port>.pid`,
log `~/fusiondance-data/logs/tracker-<port>.log` (mode 600: Trackio prints a write token into it,
so don't paste that log anywhere). Runs log to Trackio whether or not the dashboard is up.

## 2. Dry-run first

```bash
bash experiments/launch/E3.sh --dry-run
```

```powershell
.\tools\launch_experiment.ps1 E3 -DryRun
```

This prints the preflight results, the exact command, env, working directory, log path and PID
file, and executes nothing. Exit code 0 means a real launch would start; 3 means it would refuse.

Preflight checks (a `FAIL` blocks the launch, a `WARN` doesn't):

- venv python exists and `import torch, gogeta` works; CUDA available (real launches only)
- the dataset's `meta.json` exists
- the experiment's `prediction` in the registry is non-empty (no prediction, no run)
- code paths (`gogeta/ configs/ tools/ tests/ requirements.txt`) are committed (`--require-clean` would refuse anyway; untracked files count)
- no other training run is active (launcher PID files, plus any `gogeta.train` / `run_queue.sh` process)
- the GPU slot is free, no foreign GPU compute process, GPU temperature < 85 C
- the dashboard answers (WARN only), the registry status (WARN if the experiment is already `done`)

## 3. Launch

```bash
bash experiments/launch/E3.sh                  # seed from the config
bash experiments/launch/E3.sh --seed 1         # extra seeds (headline claims need >= 3)
bash experiments/launch/E3.sh --seed 2 --wait-cool
bash experiments/launch/E3.sh --foreground     # attached; Ctrl-C stops it and frees the GPU slot
```

```powershell
.\tools\launch_experiment.ps1 E3 -Seed 1
.\tools\launch_experiment.ps1 E3 -Seed 2 -WaitCool
```

Other options: `--no-compile`, `--max-tokens N` (a shorter run than the registered config; it is
flagged as a WARN). The run is started with `setsid nohup`, so it keeps going after the shell, the
PowerShell window or an agent session that launched it ends. The launcher waits ~8 s and reports
if the run died immediately (dirty code, bad config...), with the last log lines.

Files per run (`<run>` = `<EID>-s<seed>-<YYYYmmdd-HHMMSS>`):

- log: `~/fusiondance-data/logs/<run>.log`
- `~/fusiondance-data/run/train-<run>.pid` (wrapper PID = process group of the run), `.info` (command, log), `.exit` (exit code, once finished)
- the usual run folder from `gogeta.tracking`: `~/fusiondance-data/runs/<EID>-<timestamp>-s<seed>/`

Then set the experiment's `status` to `running` in `experiments/experiments.json` (AGENTS.md 5).

## 4. Follow, check, stop

```bash
tail -f ~/fusiondance-data/logs/E3-s1-<timestamp>.log    # the launcher prints the exact command
bash experiments/launch/status.sh                        # GPU slot + temperature + GPU processes, runs, CPU jobs n/8, dashboard
bash experiments/launch/stop.sh E3                       # SIGTERM to the run's process group (SIGKILL after 20 s)
bash experiments/launch/stop.sh E3-s1-20261122-183000    # a specific run, if several match
```

From PowerShell: `wsl -d Ubuntu-24.04 --cd C:\Users\mreca\Desktop\Gogeta-0 -- bash experiments/launch/status.sh`.

`status.sh` marks GPU processes that did not come through the launcher as
"NOT via the launcher": they still count against the 1-GPU-job budget, and launches wait for them.

## 5. CPU-only jobs (checks, side quests, analysis)

```bash
bash tools/launch_cpu_job.sh --name k4 -- ~/.venvs/gogeta/bin/python experiments/checks/k4_ternary_gap_vs_size.py
bash tools/launch_cpu_job.sh --dry-run -- python3 -c 'print(42)'
bash experiments/launch/stop.sh cpu:k4                   # stop CPU job(s) whose name starts with k4
```

```powershell
.\tools\launch_cpu_job.ps1 -Name k4 -- /root/.venvs/gogeta/bin/python experiments/checks/k4_ternary_gap_vs_size.py
```

Detached, from the repo root, with the CPU-only env above. Files in `~/fusiondance-data/run/cpu/`:
`<name>-<timestamp>.pid / .log / .info / .exit`. GPU training (`gogeta.train`) is refused here:
it must go through `launch_experiment.sh`.

## Caveats

- Detached runs survive the session that started them, but not `wsl --shutdown`, a Windows reboot
  or sleep that stops the WSL VM. Check `status.sh` after the laptop wakes up.
- `tools/run_queue.sh` predates these launchers: it neither takes the GPU slot nor writes PID
  files. The launchers detect its processes and refuse to start alongside them, but prefer
  `E<n>.sh` for new runs.
