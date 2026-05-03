# GlucoTwin

**GlucoTwin** is our open glucose-forecasting project: a hybrid model (physiology-informed trajectory plus learned residual and multimodal fusion) and a benchmark harness (`run_all.py`) that writes clear JSON metrics. This repository is what we want the community to see—**no** steward XML, **no** raw traces in the tree.

Optional **`example_results/`** holds small aggregate JSON and a run manifest from one example run (for format comparison only). Your own `outputs/results/` after you run the code is what you should trust for numbers.

## Quick start

1. Install [PyTorch](https://pytorch.org), then `pip install -r requirements.txt`.
2. Run `./VERIFY_PUBLIC.sh` (quick synthetic smoke with `GLUCOTWIN_EXPERIMENT_ONLY=1`).
3. For a full run or official XML, read `CONTRIBUTING.md` and `docs/OHIO_XML_SETUP.md`.

## Outputs

Metrics appear under `outputs/results/` (`run_manifest.json`, `aggregate_results.json`, …). With `GLUCOTWIN_EXPERIMENT_ONLY=1` there are no extra prediction archives or plotting steps.

## GitHub

This folder is the **root** of the GlucoTwin GitHub repository. If the hosting site suggests creating a README with `echo`, skip that—this tree already has one. Then use the usual `git init`, `git add .`, `git commit`, `git remote add origin …`, `git push` flow from **here**.

Do not commit steward XML, passwords, or raw CGM exports.
