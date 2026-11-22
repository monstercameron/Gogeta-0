#!/usr/bin/env bash
# E10: Reference control (Hugging Face LlamaForCausalLM at E0's shapes)
# Config: configs/e10_llama_ref.json   (looked up in experiments/experiments.json by the launcher)
#
#   bash experiments/launch/E10.sh [--seed N] [--dry-run] [--foreground] [--no-compile] [--max-tokens N] [--wait-cool]
#
# All checks (prediction written, clean code, dataset, GPU slot, GPU temperature...) live in
# tools/launch_experiment.sh, so every experiment starts the same way and logs to Trackio.
# LR sweeps (K7) and tuned-LR runs pass --lr through tools/launch_campaign.sh (needs the
# launcher's --lr pass-through; see experiments/launch/campaigns/README.md).
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E10 "$@"
