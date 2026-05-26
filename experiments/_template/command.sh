#!/usr/bin/env bash
# Record the exact command used for this experiment.
# Update arguments before running.

set -euo pipefail

# TODO(fas2h): replace this placeholder command with the exact run command.
python scripts/run_attack.py --config-name config attack=fas2h_mvp model=clip_3surrogate data=toy_pairs
