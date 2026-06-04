# Protein Secondary Structure Prediction with ESM-2 Embeddings

An encoder-only Transformer for **per-residue protein secondary structure prediction (PSSP)**,
built on top of frozen [ESM-2](https://github.com/facebookresearch/esm) protein language model
embeddings. The pipeline supports both **Q3** (Helix / Strand / Coil) and **Q8** (full DSSP)
classification, seed ensembling, and two layers of post-processing: a windowed **Random Forest**
smoother and the external/empirical rules of **Salamov and Solovyev**.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
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
| `export_to_excel.py` | Scores every `*_pred.fasta` / `*_true.fasta` pair with the SOV_refine Perl program (parallelised) and writes a styled Q3/Q8 Excel report. |

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

### Scoring with `export_to_excel.py`

`export_to_excel.py` automates scoring across a whole directory of predictions. It discovers
every `{name}_pred.fasta` / `{name}_true.fasta` pair, runs each one through the **SOV_refine**
Perl program in parallel (one thread per pair), and collects the parsed metrics — overall
accuracy, `SOV_99`, `SOV_refine`, and their per-class breakdowns — into a single styled
spreadsheet. Each pair becomes one row (labelled by its file name), values are reported as
percentages, and the best result per metric is highlighted.

```bash
python export_to_excel.py        # writes results.xlsx
```

Set `Q=3` or `Q=8` in the `__main__` block to switch between the 3-state and 8-state column
layouts. By default the script expects the Perl program at `SOV_refine/SOV_refine.pl` and the
FASTA pairs under `SOV_refine/predictions/`.

> **Requirement:** this step needs a working **Perl** interpreter and the **SOV_refine** package
> (`SOV_refine.pl` plus its helper files), which is *not* bundled with this repository. Obtain it
> from the SOV_refine distribution (Liu & Wang, 2018 — see Acknowledgements) and place it under
> `SOV_refine/`.

---

## Acknowledgements

-	Z. Lin *et al.*, ‘Evolutionary-scale prediction of atomic-level protein structure with a language model’, *Science*, vol. 379, no. 6637, pp. 1123–1130, Mar. 2023, doi: 10.1126/science.ade2574.

-	A. A. Salamov and V. V. Solovyev, ‘Prediction of Protein Secondary Structure by Combining Nearest-neighbor Algorithms and Multiple Sequence Alignments’, *J. Mol. Biol.*, vol. 247, no. 1, pp. 11–15, Mar. 1995, doi: 10.1006/jmbi.1994.0116.

-	M. S. Klausen *et al.*, ‘NetSurfP-2.0: Improved prediction of protein structural features by integrated deep learning’, *Proteins Struct. Funct. Bioinforma.*, vol. 87, no. 6, pp. 520–527, Jun. 2019, doi: 10.1002/prot.25674.

-	M. H. Høie *et al.*, ‘NetSurfP-3.0: accurate and fast prediction of protein structural features by protein language models and deep learning’, *Nucleic Acids Res.*, vol. 50, no. W1, pp. W510–W515, Jul. 2022, doi: 10.1093/nar/gkac439.

-	G. Wang and R. L. Dunbrack Jr, ‘PISCES: a protein sequence culling server’, *Bioinformatics*, vol. 19, no. 12, pp. 1589–1591, Aug. 2003, doi: 10.1093/bioinformatics/btg224.

-	J. A. Cuff and G. J. Barton, ‘Evaluation and improvement of multiple sequence methods for protein secondary structure prediction’, *Proteins Struct. Funct. Bioinforma.*, vol. 34, no. 4, pp. 508–519, 1999, doi: 10.1002/(SICI)1097-0134(19990301)34:4<508::AID-PROT10>3.0.CO;2-4.

-	J. Moult, K. Fidelis, A. Kryshtafovych, T. Schwede, and A. Tramontano, ‘Critical assessment of methods of protein structure prediction (CASP)—Round XII’, *Proteins Struct. Funct. Bioinforma.*, vol. 86, pp. 7–15, Dec. 2017, doi: 10.1002/prot.25415.

- T. Liu and Z. Wang, ‘SOV_refine: A further refined definition of segment overlap score and its significance for protein structure similarity’, *Source Code Biol. Med.*, vol. 13, no. 1, p. 1, Apr. 2018, doi: 10.1186/s13029-018-0068-7.

---

## License

Released under the MIT License —  See [`LICENSE`](LICENSE).

Copyright (c) 2026 Stefanos Englezou