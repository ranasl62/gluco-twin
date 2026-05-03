#!/usr/bin/env bash
# Minimal benchmark smoke: synthetic data, 2 subjects, 1 epoch, no ablations / no extra horizons.
set -eu
cd "$(dirname "$0")"
export PYTHONPATH=.
export GLUCOTWIN_EXPERIMENT_ONLY=1
export DATA_SOURCE=synthetic
export N_PATIENTS=2
export EPOCHS=1
export RUN_ABLATIONS=0
export SKIP_MULTI_HORIZON=1
export EXTEND_BASELINE_HORIZONS=0
python3 src/experiments/run_all.py
echo "OK: synthetic smoke finished; inspect outputs/results/*.json"
