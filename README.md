# Protein Secondary Structure Prediction with ESM-2 Embeddings

An encoder-only Transformer for **per-residue protein secondary structure prediction (PSSP)**,
built on top of frozen [ESM-2](https://github.com/facebookresearch/esm) protein language model
embeddings. The pipeline supports both **Q3** (Helix / Strand / Coil) and **Q8** (full DSSP)
classification, seed ensembling, and two layers of post-processing: a windowed **Random Forest**
smoother and the classic **Salamov–Solovyev** structural rules.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
</p>

---

## Overview

The model treats each residue independently. A single ESM-2 embedding (dimension `1280`) is
sliced into a fixed number of equally sized **tokens**, given sinusoidal positional encodings,
and passed through a stack of standard Transformer encoder layers (multi-head self-attention +
FFN). The token representations are mean-pooled and classified by an MLP head.

```
Sequence ──► ESM-2 (frozen, 650M) ──► [N, 1280] embeddings
                                          │
                                          ▼
                       slice into tokens ──► + positional encoding
                                          │
                                          ▼
                       N × Transformer encoder layers (self-attention)
                                          │
                                          ▼
                       mean-pool over tokens ──► MLP head ──► Q3 / Q8 logits
```

On top of the base network, three optional stages improve robustness:

1. **Seed ensemble** — several models trained with different seeds have their softmax
   probabilities averaged at inference (`EnsemblePSSP`).
2. **Random Forest post-processing** — a sliding window of ensemble probabilities
   (default width `31`) is fed to a Random Forest that refines the per-residue call.
3. **External rules (Q3 only)** — the Salamov–Solovyev heuristics remove implausibly short
   helices/strands. These can be combined with the RF stage in either order (`ER→RF`, `RF→ER`).

---

## Repository structure

| File | Purpose |
|------|---------|
| `config.json` | All hyperparameters, paths, dataset definitions, and run settings. |
| `config.py` | Singleton loader exposing `config.json` as a dict-like object. |
| `data_manager.py` | Parses raw datasets and extracts frozen ESM-2 embeddings into compressed `.npz` archives. |
| `pssp_encoder.py` | Model definitions: `PSSPEncoder`, encoder `Layer`, `PositionalEncoding`, and `EnsemblePSSP`. |
| `preprocessing.py` | DSSP ↔ integer label mappings and the Q8→Q3 reduction table. |
| `train_utils.py` | Training loop, batching, optimizer/scheduler factory, seeding, checkpointing, curve logging. |
| `optuna_tuning.py` | Hyperparameter search (Optuna TPE sampler + Median pruner, persisted to SQLite). |
| `cross_validation.py` | k-fold × multi-seed cross-validation, plus ensemble and RF FASTA generation. |
| `final_training.py` | Full-data training on all folds and evaluation-set inference (base / RF / ER variants). |
| `external_rules.py` | Salamov–Solovyev Q3 structural post-processing rules. |
| `save_predictions.py` | Logit extraction, windowed feature construction, and FASTA writers. |

---

## Installation

Requires **Python 3.11.13** and a CUDA 12.8 capable GPU (the pinned PyTorch build is
`torch==2.7.1+cu128`).

```bash
git clone https://github.com/seggle01/PSSP-Implementation-BSc-Thesis.git
cd PSSP-Implementation-BSc-Thesis
```

### Option A — Conda (recommended)

Conda pins the exact Python version and keeps the environment fully isolated:

```bash
conda create -n pssp python=3.11.13
conda activate pssp

pip install -r requirements.txt
```

### Option B — venv

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** the CUDA-enabled PyTorch wheel (`torch==2.7.1+cu128`) is pulled from the PyTorch
> wheel index declared at the top of `requirements.txt`, so `pip install -r requirements.txt`
> works the same under either option.

Set `"DEVICE": "cpu"` in `config.json` to run without a GPU, though embedding extraction will
be slow.

---

## Data format

Input datasets are plain-text, FASTA-like files with **name / sequence / target** triplets:

```
>1abcA
MKTAYIAKQR...
CCCHHHHHEEE...
```

Place them under the directory tree expected by `config.json`, e.g.:

```
data/
├── PISCES_3class/CASP13.txt
├── NetsurfP_3class/CB513.txt
└── NetsurfP_8class/CB513.txt
```

The state (`3` or `8`) is inferred from the `_3class` / `_8class` affix in the path.

---

## Usage

### 1. Extract embeddings

Edit the `datasets` list in `data_manager.py`, then run:

```bash
python data_manager.py
```

This loads the frozen ESM-2 model, embeds every residue, and writes
`{name}_{state}class.npz` archives (embeddings, labels, chain IDs) to the configured
`embed_dir`. Fold files are expected to follow the `{fold_prefix}{i}_{state}.npz` naming
convention (e.g. `train_fold0_3class.npz`).

### 2. (Optional) Tune hyperparameters

```bash
python optuna_tuning.py
```

Runs an Optuna study per configuration in `config.json → TUNING`, searching over layers,
heads, dropout, learning rate, and more. Studies are resumable and stored as SQLite databases
under `optuna_results/`. Copy the best parameters back into `MODEL_CONFIG` in `config.json`.

### 3. Cross-validation

```bash
python cross_validation.py
```

Runs three stages (toggle via the `codes` list in `main()`):

1. **Nested k-fold × multi-seed CV** of the base model.
2. **Seed-ensemble inference** → per-chain prediction/truth FASTA files.
3. **Random Forest post-processing** trained per fold → RF FASTA files.

### 4. Final training & evaluation

```bash
python final_training.py
```

Trains final models on all folds combined (one per seed) and evaluates on the held-out test
sets (e.g. `CB513`, `CASP12`, `CASP13`, `TS115`). The four stages produce, respectively:
final model checkpoints, base ensemble FASTA, a Random Forest post-processor, and the
RF / ER / ER→RF / RF→ER prediction variants.

---

## Configuration

Everything is driven by `config.json`, loaded through a singleton `Config` object:

```python
from config import Config
cfg = Config()

cfg['DEVICE']                       # "cuda"
cfg['MODEL_CONFIG']['3class_params']
cfg['WINDOW_SIZE']                  # 31
```

Key sections:

- **`ESM2`** — model name and embedding dimension.
- **`MODEL_CONFIG`** — separate architecture/optimization blocks for `3class` and `8class`.
- **`TRAINING_CONFIG`** — split ratio, epochs, batch size, optimizer betas, LR floor.
- **`TUNING` / `CV_RUNS` / `FINAL_TRAINING`** — per-experiment fold counts, patience, and
  evaluation sets.
- **`RF_CONFIG`** — Random Forest hyperparameters.
- **`USE_8CLASS`** — train on 8-class fold files and reduce to Q3 at load time, allowing a
  3-class model to benefit from the finer-grained labels.

---

## Model details

- **Architecture:** encoder-only Transformer with sinusoidal positional encodings, post-norm
  encoder layers, and a GELU MLP classification head.
- **Backbone:** `facebook/esm2_t33_650M_UR50D` (frozen — no fine-tuning of the PLM).
- **Optimization:** AdamW (β₁ = 0.9, β₂ = 0.95) with cosine-annealing LR, gradient clipping,
  and early stopping on validation loss.
- **Ensembling:** logits averaged across models trained with seeds `42`, `123`, and `1337`.

---

## Evaluation

Predictions are written as paired FASTA files (`*_pred.fasta` / `*_true.fasta`), one record per
chain, which makes them straightforward to score with any standard Q3/Q8 accuracy or SOV
metric of your choice.

---

## Acknowledgements

- ESM-2 protein language model — Meta AI (FAIR).
- Salamov & Solovyev — secondary structure prediction post-processing rules.
- Benchmark datasets: NetSurfP, PISCES, and the CASP/CB513/TS115 evaluation sets.

---

## Author

**Stefanos Englezou**
