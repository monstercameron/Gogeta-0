#!/usr/bin/env bash
# E8: Self-speculative decoding
# NOT DESIGNED YET: no config and no prediction in experiments/experiments.json.
#
# To make this launchable: add configs/e8_<name>.json, set "config" and write the
# "prediction" for E8 in the registry (commit both before the first run, AGENTS.md 6), then
# replace the two lines below with the standard body used by E3.sh:
#   exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E8 "$@"
echo "E8 (Self-speculative decoding) is not designed yet: add a config and a prediction first (experiments/experiments.json, AGENTS.md 6). Nothing was launched." >&2
exit 3
