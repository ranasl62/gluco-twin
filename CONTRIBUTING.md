# Contributing to GlucoTwin

Thank you for looking at **GlucoTwin**. This repo is the part we share openly: the model, the benchmark, and how to run it responsibly with Ohio data if you have steward access.

## What is in this repo

- `src/glucotwin/` — The model (`model.py`, `tft.py`, `bergman.py`, …), synthetic cohort helpers, XML parsing (`ohio_xml.py`), and loaders for official data.
- `src/experiments/` — The main entry point `run_all.py`.
- `example_results/` — Small optional JSON samples (aggregates and a manifest, not CGM traces).
- `docs/OHIO_XML_SETUP.md` — Where to put official XML on your machine and which flags matter.

## How to run the benchmark

We ship runs with **`GLUCOTWIN_EXPERIMENT_ONLY=1`** so you get **JSON metrics** without extra side artifacts. `./VERIFY_PUBLIC.sh` uses that for a short synthetic check.

Synthetic (no steward files needed):

```bash
export GLUCOTWIN_EXPERIMENT_ONLY=1
PYTHONPATH=. python3 src/experiments/run_all.py
```

Official OhioT1DM XML (only if you already obtained files under steward terms):

```bash
export GLUCOTWIN_EXPERIMENT_ONLY=1
DATA_SOURCE=real OHIO_ROOT=data/OhioT1DM OHIO_SKIP_TEST_HEAD_STEPS=12 \
  PYTHONPATH=. python3 src/experiments/run_all.py
```

## Reproducibility

If you change the model or the protocol, run `run_all.py` again so your saved JSON matches what you describe.

## Data

OhioT1DM XML is steward-restricted. Do not commit raw XML, passwords, or download links here. Read `docs/OHIO_XML_SETUP.md` and `data/OhioT1DM/README.md`.

## Docker

The `Dockerfile` follows the same path as `run_all.py` if you prefer a container.

## Git author (avoid “Cursor Agent” on GitHub)

GitHub’s **Contributors** list comes from **commit author name and email**. If your editor created the first commit with a default such as **Cursor Agent** / **cursoragent**, that identity can appear beside your own.

1. **Use your identity in this repo** (already set in `.git/config` for maintainers; repeat after a fresh clone if needed):

   ```bash
   git config user.name "Your Real Name"
   git config user.email "your.email@institution.edu"
   ```

2. **Before pushing**, this repo’s `.githooks/pre-push` rejects commits whose author/email match common Cursor automation patterns.

3. **If the wrong name is already on GitHub**, rewrite history and force-push (only if you control the repo and collaborators agree), e.g. with [git-filter-repo](https://github.com/newren/git-filter-repo) to set author/committer on all commits, or amend the latest commit with `git commit --amend --reset-author` and force-push when it is safe to do so.

4. **Confirm locally** before pushing: `git log --format='%an <%ae>' | sort -u` should list only humans you expect.
