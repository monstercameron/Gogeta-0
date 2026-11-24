# Fleet and version tooling (saved from the 2026-11-23 session)

These scripts drove every rented-GPU batch of the assembly phase (A1-A4, B1, C1) and cut every version from v0.5.x on. They lived in a session scratchpad and are saved here so the next session can reuse them. **No secrets are in them.** The Vast API key lives only in the vastai CLI config and in a file outside the repo; the SSH key is `/root/.ssh/gogeta_vast` in WSL and never leaves the machine. Never put either in the repo or pass them to another agent or service.

## Paths to fix before reuse

The scripts hard-code the old scratchpad as their working directory, in the variables `S`, `WS`, `SW`, `GB` and `W`, plus `$S/c1/...`. Pick a new working directory outside the repo, for example `C:\Users\mreca\gogeta-fleet`, which is `/mnt/c/Users/mreca/gogeta-fleet` in WSL. Copy `tools/fleet/*` and the `wave/` folder there, rename `wave/` to the new batch name (for example `c2/`), and replace the old path and the `/c1/` and `-c1-` names, for example with `sed`. The box label must start with `gogeta-cockpit-`, because `destroy_box.ps1` refuses any other label.

## One GPU batch, end to end

1. **Register first** (AGENTS rules and `experiments/checks/PREDICTIONS.md`): configs, the predictions, the decision rule, and the readout script's sha256-16. Cut and push that version. Dispatch only from the pushed commit (`SHA=$(git rev-parse vX.Y.Z^{commit})`).
2. **Check that nothing is already rented**, then search and rent one box:
   `vastai search offers 'gpu_name=RTX_5090 num_gpus=1 reliability>0.98 verified=true rentable=true direct_port_count>=1 disk_space>=60 inet_down>=500' -o dph_total`
   `vastai create instance <offer> --image pytorch/pytorch:latest --disk 60 --ssh --direct --label gogeta-cockpit-<wave>-1 --raw` (see `rent_example.ps1`).
3. `wave/waitup.sh` polls until the box is up with SSH, then writes `endpoints.txt`. Copy it to `endpoints_all.txt`.
4. Write `jobs-<label>.txt`, one job per line as `name|train args` (`wave/jobs.example.txt` is C1's list), then run `SHA=<commit> TESTS=<label> bash wave/drive.sh` in the background. It bootstraps the box at SHA, builds and verifies the code-py-v2 data against the committed manifest, optionally runs the GPU test suite and any extra checks, and launches `worker.sh`, which trains each job and then scores every checkpoint on clean80.
5. Run **one** controller per fleet, `SHA=<commit> bash wave/dispatch.sh`, in the background. Every 60 seconds it checks the box. When the worker reports JOBS DONE, it pulls logs, runs and scores (and new checkpoints) through `pull.sh`, then assigns jobs from `queue.txt` or destroys the box. A `STOP` file in the wave folder makes it exit without touching boxes. After any stop, check with `ps` that no orphan controller is left (it happened once).
6. `wave/peek.sh` shows progress. `pull.sh <label>` pulls by hand.
7. **After teardown:** run `vastai show instances` and expect 0. Reconcile spend from the provider's credit before and after into `experiments/expenses.json`.

Known traps, each of which cost something:
- **Checkpoint date filter:** `pull.sh` once matched only `-20261123-` checkpoints. After midnight UTC the runs were named `-20261124-` and were not pulled before the box was destroyed, so batch C1's F-6 checkpoints were lost. It now matches `-2026112[3-9]-`; widen it again for later dates.
- **Repo dependencies:** the box needs `remote_v2build_sync.sh` and the committed v2 manifest, and `assign.sh` refuses to run if the data hash differs.
- **Light work only on the N1X** (operator hold since a thermal kill). Rented boxes do the GPU work.

## Cutting a version

`python tools/fleet/cut_version.py spec.json`, where spec.json holds `{"version", "date", "title", "entry", "body", "files", "site", "index"}`. It bumps pyproject and the README badge and status prefix, prepends the DEVLOG entry, regenerates the claims list, site data and experiment index, commits, makes the annotated tag, and pushes. `files` lists the paths to stage. Never stage another agent's work-in-progress files.

## Other helpers

- `runtests.py` runs module-level `test_*` functions. It shows 0/0 for unittest-class files such as `test_moe.py`; run those with `python -m unittest`.
- `destroy_box.ps1 -Label <label>` destroys exactly one labelled box and refuses ambiguous matches.
