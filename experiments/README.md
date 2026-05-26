# Experiment Management

This repository keeps one shared `src/` codebase and stores experiment-specific state under `experiments/`.
The goal is to keep runs reproducible without copying or forking algorithm code for every trial.

## Directory Roles

- `active/`: experiments that are being run, debugged, or analyzed now.
- `archive/`: experiments with results worth keeping for comparison or reporting.
- `graveyard/`: failed, invalid, or intentionally abandoned experiments that should still keep their audit trail.
- `_template/`: starter files for new experiment folders.

## Required Files Per Experiment

Each experiment directory should contain:

- `config.yaml`: copy of the config used at creation time.
- `command.sh`: exact or intended run command.
- `git_commit.txt`: commit id used for the experiment.
- `data_version.txt`: dataset snapshot or version note.
- `README.md`: goal, key changes, and current conclusion.
- `failures.md`: failed attempts and debugging notes.
- `summary.json`: concise machine-readable summary with metrics and status.
- `logs/`: log files that stay local and are ignored by Git.

Optional large outputs such as `checkpoints/`, `metrics.csv`, `captions.jsonl`, and `judge_scores.jsonl` may exist in the same experiment directory, but they should not be committed unless intentionally curated.

## Naming Convention

Use `YYYYMMDD_<short_name>` for experiment directories, for example:

`20260526_fas2h_clip3_topk32_seed42`

Recommended `<short_name>` fields:

- method or branch name
- important ablation knob
- seed

Avoid opaque names such as `test3` or `retry_new`.

## Failure Recording

Failed experiments still belong in version control if the metadata is useful.

- Keep the folder.
- Write the failure in `failures.md`.
- Add a short `failure_reason` in `summary.json`.
- Move the folder to `graveyard/` when the run is no longer active.

## Three-Month Repro Rule

To make an experiment reproducible after three months, record at least:

- copied config
- exact command
- git commit id
- data version or dataset snapshot note
- random seed
- final metric summary
- failure notes if the run did not complete

If dependencies or environment differ from the repo default, note that in the experiment `README.md`.
