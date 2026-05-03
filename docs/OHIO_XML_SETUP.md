# OhioT1DM XML (layout and evaluation flags)

## Directory layout

- Place official XML only under `data/OhioT1DM/{2018,2020}/{train,test}/`, using curator filenames (e.g. `540-ws-training.xml`, `540-ws-testing.xml`).

## Legal / sharing

- Raw XML is restricted by the steward’s terms. Never commit XML, passwords, or download URLs/tokens to a **public** git remote.
- Cite Marling & Bunescu (2020) when reporting quantitative results on this corpus.

## Code entry points

- Parser: `src/glucotwin/ohio_xml.py` (`parse_patient_xml`, `discover_pairs`, `load_train_test_patient`).
- PyTorch loaders: `create_dataloaders_real()` in `src/glucotwin/dataset.py` (training-split statistics for normalization on real data).
- Full benchmark: `DATA_SOURCE=real PYTHONPATH=. python3 src/experiments/run_all.py`, or Docker with `-e DATA_SOURCE=real`.

## Evaluation protocol flags

- **`OHIO_ROOT`** — Root folder for XML trees (default `data/OhioT1DM`).
- **`OHIO_SKIP_TEST_HEAD_STEPS`** — Default **12** (drops the first sixty minutes of five-minute **test** samples for **2020** cohort IDs only; aligns with the 2020 BGLP scoring description). Set **`0`** to score full test exports.

## EHR channels on real XML

- The public XML release does not ship full structured EHR tables. The loader uses a **47-dimensional zero placeholder** so the model’s EHR branch remains wired; interpret EHR-related ablations on real XML cautiously.
