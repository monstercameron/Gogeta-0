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
`#` starts a comment. `--lr` needs the launcher's `--lr` pass-through (proposed alongside this
folder; the runner refuses a campaign that uses `--lr` until the launcher has it). Every override
is recorded in the run's config (`train`, `cli_overrides`) and in the launcher's log header.

| File | What |
|---|---|
| `phase2a-k7-lr.campaign` | K7: 8 runs, 20M tokens each, lr in {1e-3, 2e-3, 3e-3, 6e-3} for E10 and E1. ~43 min GPU. |
| `phase2b-replication.template` | Replication at the tuned LRs (3 seeds each of E10, E1 and E11, then E9, conditional E0/E2). `experiments/checks/k7_lr_sweep.py --write-campaign` turns it into `phase2b-replication.campaign`. ~5.8 h GPU, up to ~7.5 h with the conditional runs. |
