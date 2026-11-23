# Campaigns

A campaign is a list of launches that `tools/launch_campaign.sh` runs **one after another**, each
through `tools/launch_experiment.sh --foreground --wait-cool`, so every run takes and releases the
single GPU slot and passes the normal preflight (prediction written, clean code paths, GPU idle and
below 85 C). The runner is detached (survives the session that started it), waits while anything
else uses the GPU, and logs progress to `~/fusiondance-data/logs/campaign-<name>-<stamp>.log`.

```bash
bash tools/launch_campaign.sh experiments/launch/campaigns/phase2a-k7-lr.campaign --dry-run
bash tools/launch_campaign.sh experiments/launch/campaigns/phase2a-k7-lr.campaign
bash tools/launch_campaign.sh --status            # progress, finished items, last events
bash tools/launch_campaign.sh --wait phase2a-k7-lr
bash tools/launch_campaign.sh --stop phase2a-k7-lr
```

Format: one launch per line, `<EXPERIMENT_ID> [--seed N] [--lr X] [--max-tokens N] [--no-compile]`;
`#` starts a comment. `--lr` is passed through by the launcher (QUEUE item Q2, done); the runner
refuses a campaign that uses `--lr` if the launcher lacks it. Every override
is recorded in the run's config (`train`, `cli_overrides`) and in the launcher's log header.

| File | What |
|---|---|
| `phase2a-k7-lr.campaign` | K7: 8 runs, 20M tokens each, lr in {1e-3, 2e-3, 3e-3, 6e-3} for E10 and E1. ~43 min GPU. Done. |
| `phase2a-k7-lr-resume.campaign` | K7 items 2-8, resumed after an interruption (v0.4.16). Done. |
| `phase2a-k7-lr-ext.campaign` | K7 grid extension (E1 at lr 0.012), added by the decision rule. Done. |
| `phase2b-replication.campaign` | Generated from the template with LR_E10=0.003 and LR_E1=0.006 (K7-tuned). Paused at 6 of 11 items on the N1X: E10 and E1 seeds 0–2 ran, then E11 seed 0 was killed at 86.6 °C and an operator hold (human-only release) stopped the rest (DEVLOG v0.4.73). A 5090 replication (E10 and E1 at 7 seeds each, DEVLOG v0.4.74–v0.4.79) ran on rented RTX 5090s and is complete (v0.4.79); it is separate from this file. The file's order is E10, E1, E11, E9, then E2 at 6e-3. `experiments/QUEUE.md` (last updated 2026-11-23 01:20) is the current queue. |
| `phase2b-replication-resume-1.campaign` | Watchdog resume of `phase2b-replication` (written 2026-11-22 20:53): the 8 unfinished items. Not run: Phase 2b is still paused under the operator hold (DEVLOG v0.4.73). |
| `phase2b-replication-resume-2.campaign` | Watchdog resume of resume-1 (written 21:01), with the same 8 items: resume-1 finished none of them. Not run (same hold). |
| `phase2b-replication.template` | Replication at the tuned LRs (3 seeds each of E10, E1 and E11, then E9, conditional E0/E2). `experiments/checks/k7_lr_sweep.py --write-campaign` turns it into `phase2b-replication.campaign`. ~5.8 h GPU, up to ~7.5 h with the conditional runs. |
