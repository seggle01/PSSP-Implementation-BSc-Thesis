import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from train_utils import load_npz_dataset, batch_iterator
from preprocessing import parse_int_to_q3, parse_int_to_q8, reduce_q8_to_q3
from config import Config

######### Module-level configuration #########
cfg        = Config()
device     = cfg['DEVICE']
batch_size = cfg['TRAINING_CONFIG']['batch_size']
embed_dir  = cfg['PATHS']['embed_dir']

os.makedirs(cfg['PATHS']['cv_fasta'], exist_ok=True)

######### Ensemble logit extraction (softmax probabilities, residue-flat) #########
def extract_logits(model: nn.Module, filenames: list[str], run_config_name: str):
    """
    Load the given .npz files, run inference, return flat arrays:
      logits    : [N, Q]   softmax probabilities
      labels    : [N]      int8 true labels
      chain_ids : [N]      str  residue-level chain IDs
    """
    embeddings_list, labels_list, chain_ids_list = [], [], []

    # Concatenate all requested folds into a single flat record
    for name in filenames:
        try:
            X, Y, ids = load_npz_dataset(os.path.join(embed_dir, name))
            if cfg['USE_8CLASS'] == run_config_name:
                Y = reduce_q8_to_q3(Y)
            embeddings_list.append(X)
            labels_list.append(Y)
            chain_ids_list.append(ids)
            print(f"  Loaded {name}: {len(X)} residues")
        except FileNotFoundError:
            print(f"  WARNING: {name} not found, skipping.")

    embeddings = np.concatenate(embeddings_list, axis=0)   # [N, 1280]
    labels     = np.concatenate(labels_list,     axis=0)   # [N]
    chain_ids  = np.concatenate(chain_ids_list,  axis=0)   # [N]

    # Batched ensemble inference (averaged logits → softmax)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch_emb, _ in tqdm(batch_iterator(embeddings, labels), desc="Inference", unit="batch"):
            probs = F.softmax(model(batch_emb), dim=-1)  # [B, Q]
            all_logits.append(probs.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)   # [N, Q]
    return logits, labels, chain_ids

#########  Sliding-window feature construction (chain-aware, zero-padded) #########
def build_windowed_dataset(logits: np.ndarray, labels: np.ndarray,
                           chain_ids: np.ndarray, window_size: int):
    assert window_size % 2 == 1, "window_size must be odd"
    half = window_size // 2
    N, Q = logits.shape

    # Locate chain boundaries (so windows never cross proteins)
    change_mask  = np.concatenate(([True], chain_ids[1:] != chain_ids[:-1]))
    chain_starts = np.where(change_mask)[0]
    chain_ends   = np.concatenate((chain_starts[1:], [N]))

    X_list, y_list = [], []

    # Per-chain windowing 
    for start, end in zip(chain_starts, chain_ends):
        chain_logits = logits[start:end]   # [L, Q]
        chain_labels = labels[start:end]   # [L]
        L            = end - start

        # Zero-pad on both sides: [L + 2*half, Q]
        padded = np.zeros((L + 2 * half, Q), dtype=np.float32)
        padded[half: half + L] = chain_logits

        # Vectorised sliding window → [L, window_size, Q]
        shape   = (L, window_size, Q)
        strides = (padded.strides[0], padded.strides[0], padded.strides[1])
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

        # Flatten last two dims → [L, window_size * Q]
        X_list.append(windows.reshape(L, window_size * Q).copy())  # .copy() to own memory
        y_list.append(chain_labels.astype(np.int64))

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

#########  Base-model / ensemble predictions → per-chain FASTA #########
def get_predictions_to_fasta(
        model: nn.Module,
        npz_path: str,
        out_pred_fasta: str,
        out_true_fasta: str,
        Q: int,
        run_config_name: str
    ):

    # Pick label decoder for Q3 / Q8
    if Q == 3:
        int_to_ss = parse_int_to_q3
    elif Q == 8:
        int_to_ss = parse_int_to_q8
    else:
        raise ValueError(f"Invalid Q: {Q}")

    # Load flat residue-level arrays
    embeddings, labels, chain_ids = load_npz_dataset(os.path.join(embed_dir, npz_path))

    model.eval()

    # Batch inference, argmax decoding
    all_preds = []
    with torch.no_grad():
        for batch_emb, _ in tqdm(batch_iterator(embeddings, labels), desc="Inference", unit="batch"):
            logits = model(batch_emb)
            preds  = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

    all_preds = np.array(all_preds, dtype=np.int64)  # [N]

    # Locate chain boundaries to reconstruct per-chain FASTA
    N            = len(chain_ids)
    change_mask  = np.concatenate(([True], chain_ids[1:] != chain_ids[:-1]))
    chain_starts = np.where(change_mask)[0]
    chain_ends   = np.concatenate((chain_starts[1:], [N]))

    # Write predictions + ground truth FASTA
    with open(out_pred_fasta, "w") as pred_f, open(out_true_fasta, "w") as true_f:
        for start, end in tqdm(zip(chain_starts, chain_ends), total=len(chain_starts), desc="Writing FASTA", unit="chain"):
            chain_id     = chain_ids[start]
            chain_preds  = all_preds[start:end]
            chain_labels = labels[start:end]

            pred_ss = "".join(int_to_ss(p)      for p in chain_preds)
            if cfg['USE_8CLASS'] == run_config_name:
                true_ss = "".join(int_to_ss(reduce_q8_to_q3(int(l))) for l in chain_labels)
            else:
                true_ss = "".join(int_to_ss(int(l)) for l in chain_labels)

            pred_f.write(f">{chain_id}\n{pred_ss}\n")
            true_f.write(f">{chain_id}\n{true_ss}\n")

    print(f"Predictions written → {out_pred_fasta}")
    print(f"True labels written → {out_true_fasta}")

#########  Random Forest predictions → per-chain FASTA #########
def get_rf_predictions_to_fasta(
        rf,                      # trained RandomForestClassifier
        logits: np.ndarray,      # [N, Q]  softmax probs from ensemble
        labels: np.ndarray,      # [N]     true labels
        chain_ids: np.ndarray,   # [N]     residue-level chain IDs
        window_size: int,
        out_pred_fasta: str,
        out_true_fasta: str,
        Q: int
    ):
    # Pick label decoder
    if Q == 3:
        int_to_ss = parse_int_to_q3
    elif Q == 8:
        int_to_ss = parse_int_to_q8
    else:
        raise ValueError(f"Invalid Q: {Q}")

    N    = len(chain_ids)
    half = window_size // 2

    # Chain boundary detection
    change_mask  = np.concatenate(([True], chain_ids[1:] != chain_ids[:-1]))
    chain_starts = np.where(change_mask)[0]
    chain_ends   = np.concatenate((chain_starts[1:], [N]))

    os.makedirs(os.path.dirname(out_pred_fasta) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_true_fasta) or ".", exist_ok=True)

    # Per-chain windowing + RF prediction + FASTA write
    with open(out_pred_fasta, "w") as pred_f, open(out_true_fasta, "w") as true_f:
        for start, end in tqdm(
            zip(chain_starts, chain_ends),
            total=len(chain_starts),
            desc="Writing RF FASTA", unit="chain"
        ):
            chain_logits = logits[start:end]     # [L, Q]
            chain_labels = labels[start:end]     # [L]
            L            = end - start
            chain_id     = chain_ids[start]

            # Build windowed features for this chain
            padded = np.zeros((L + 2 * half, logits.shape[1]), dtype=np.float32)
            padded[half: half + L] = chain_logits

            shape   = (L, window_size, logits.shape[1])
            strides = (padded.strides[0], padded.strides[0], padded.strides[1])
            X_chain = np.lib.stride_tricks.as_strided(
                padded, shape=shape, strides=strides
            ).reshape(L, -1).copy()              # [L, window_size * Q]

            preds = rf.predict(X_chain)          # [L]

            pred_ss = "".join(int_to_ss(p)      for p in preds)
            true_ss = "".join(int_to_ss(int(l)) for l in chain_labels)

            pred_f.write(f">{chain_id}\n{pred_ss}\n")
            true_f.write(f">{chain_id}\n{true_ss}\n")

    print(f"RF Predictions written → {out_pred_fasta}")
    print(f"RF True labels written → {out_true_fasta}")
