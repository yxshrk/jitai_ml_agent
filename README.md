# TikTok TechJam 2026 - Project 2

This repository starts from the organizer-provided **KuaiRand-Pure Starter Kit** for the "Autonomous Machine Learning Research Agent for Recommender Systems" track.

The fixed benchmark contract is:

- train only on the provided KuaiRand-Pure data;
- develop against the validation split, not the hidden final-evaluation data;
- rank logged impressions within each user using `long_view` as the relevance label;
- keep the organizer's `evaluate.py` unchanged; the primary validation score is the mean of GAUC and nDCG@5.

## Quick start

Requires Python 3.9 or newer and a shell with `curl` and `tar`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
./scripts/bootstrap_data.sh
cd kuairand-starter-kit
python baseline.py --model fm
```

The public data download is about 45 MB, is stored under `kuairand-starter-kit/KuaiRand-Pure/`, and is intentionally ignored by Git.

The organizer's baseline also prints a score for its bundled public test split. Do not use that output to select changes: iterate and compare models using validation results only.

## Important brief discrepancy

One line in the supplied PDF names a different task (`click`, NDCG@10, and Recall@50). The starter kit's executable evaluator and the later starter-kit section of the brief instead specify `long_view`, GAUC, and nDCG@5. This repository follows the executable `evaluate.py` contract; confirm the discrepancy with the organizers before final submission rather than changing the evaluator.

## Starter-kit commands

Run these from `kuairand-starter-kit/` after downloading the data:

```bash
# Official NumPy Factorization Machine baseline.
python baseline.py --model fm

# Harness sanity checks.
python baseline.py --model random
python baseline.py --model pop

# Make and validate an example submission.
python submit.py --make --split valid example_submission.csv
python submit.py --check --split valid example_submission.csv
python submit.py --score --split valid example_submission.csv
```

The source kit is deliberately minimal: Python plus NumPy, with fixed data splitting and evaluation in `kuairand-starter-kit/data.py` and `kuairand-starter-kit/evaluate.py`. Preserve those scoring conventions while the research agent iterates on features, objectives, models, or training strategy. Record every experiment's hypothesis, code change, metrics, and recovery events for the competition's autonomy and robustness requirements.
