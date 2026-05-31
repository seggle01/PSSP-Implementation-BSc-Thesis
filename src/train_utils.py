import torch
import torch.nn as nn
import os, json, random
import optuna
import numpy as np
from src.config import Config
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.preprocessing import reduce_q8_to_q3
from tqdm import tqdm

######### Module-level configuration #########
cfg        = Config()
device     = cfg['DEVICE']
batch_size = cfg['TRAINING_CONFIG']['batch_size']

os.makedirs(cfg['PATHS']['cv_models'],       exist_ok=True)
os.makedirs(cfg['PATHS']['training_curves'], exist_ok=True)

######### Reproducibility #########
def set_seed(seed=None):
    if seed is None:
        raise ValueError("A seed must be explicitly provided for reproducibility.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

######### Fold concatenation helper #########
def concat_folds(loaded_folds: dict, fold_names: list):
    """Concatenate X, Y arrays from a subset of loaded folds."""
    X = np.concatenate([loaded_folds[n][0] for n in fold_names], axis=0)
    Y = np.concatenate([loaded_folds[n][1] for n in fold_names], axis=0)
    return X, Y

######### Optimiser / scheduler / loss factory #########
def build_optimizer_scheduler(model, params: dict, epochs: int):
    """Build AdamW + CosineAnnealingLR + CrossEntropyLoss from a params dict."""
    # AdamW with β1=0.9, β2=0.95
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = params['learning_rate'],
        weight_decay = params['weight_decay'],
        betas        = (cfg['TRAINING_CONFIG']['beta_1'], cfg['TRAINING_CONFIG']['beta_2'])
    )
    # Cosine annealing without restarts
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg['TRAINING_CONFIG']['eta_min'])
    loss_fn   = nn.CrossEntropyLoss()
    return optimizer, scheduler, loss_fn

######### NPZ dataset loading #########
def load_npz_dataset(npz_path: str):
    """Load a saved npz dataset into memory."""
    data       = np.load(npz_path)
    embeddings = data['embeddings']   # [N, 1280] float32
    labels     = data['labels']       # [N] int8
    chain_ids  = data['chain_ids']    # [N] str
    data.close()
    return embeddings, labels, chain_ids

def load_folds(fold_names: list, embed_dir: str, use_8class: bool = False):
    """
    Load a list of .npz fold files, optionally remapping Q8→Q3.
    Returns a dict {name: (X, Y, ids)} and skips missing files with a warning.
    """
    loaded = {}
    for name in fold_names:
        try:
            X, Y, ids = load_npz_dataset(os.path.join(embed_dir, name))
            if use_8class:
                Y = reduce_q8_to_q3(Y)
            loaded[name] = (X, Y, ids)
            print(f"  {name}: {len(X):,} residues")
        except FileNotFoundError:
            print(f"  WARNING: {name} not found, skipping.")
    return loaded

######### Batch iteration over residue-level NPZ arrays #########
def fetch_batch_from_npz(offset, embeddings, labels, total_samples):
    # Exhausted
    if offset >= total_samples:
        return None, None

    # Slice next batch and transfer to GPU
    end_offset = min(offset + batch_size, total_samples)
    batch_embeddings = torch.from_numpy(embeddings[offset:end_offset]).float().to(device)
    batch_labels     = torch.from_numpy(labels[offset:end_offset].astype(np.int64)).to(device)
    return batch_embeddings, batch_labels

def batch_iterator(embeddings, labels):
    total  = len(embeddings)
    offset = 0
    while offset < total:
        batch_embeddings, batch_labels = fetch_batch_from_npz(offset, embeddings, labels, total)
        if batch_embeddings is None or batch_embeddings.size(0) == 0:
            break
        yield batch_embeddings, batch_labels
        offset += batch_embeddings.size(0)

######### Single train / validation phase #########
def run_phase(model, dataloader_fn, loss_fn, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        with tqdm(dataloader_fn(), desc=desc, unit="batch", dynamic_ncols=True) as pbar:
            for inputs, targets in pbar:
                # Forward (PSSPEncoder returns (logits, attn_weights))
                logits = model(inputs)[0]
                loss   = loss_fn(logits, targets)

                # Backward + clipped step
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Running metrics
                batch_loss  = loss.item()
                total_loss += batch_loss * logits.size(0)
                correct    += (logits.argmax(-1) == targets).sum().item()
                total      += logits.size(0)

                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{correct/total:.4f}")
                del logits, loss

    return total_loss / total, correct / total

######### Training loop with early stopping + best-loss checkpointing #########
def train(model, optimizer, scheduler, loss_fn,
          train_data, val_data,
          epochs, patience, model_dir=None,
          save_name=None, trial=None):

    X_train, Y_train = train_data
    X_val,   Y_val   = val_data

    best_val_acc, min_val_loss     = 0.0, float('inf')
    patience_counter, best_epoch   = 0, 0
    train_losses, val_losses       = [], []
    train_accuracies, val_accuracies = [], []

    for e in range(epochs):
        # Training epoch
        def train_loader():
            return batch_iterator(X_train, Y_train)
        train_loss, train_acc = run_phase(
            model, train_loader, loss_fn, optimizer,
            desc=f"Epoch {e+1}/{epochs} Train"
        )
        scheduler.step()

        # Validation epoch
        def val_loader():
            return batch_iterator(X_val, Y_val)
        val_loss, val_acc = run_phase(
            model, val_loader, loss_fn,
            desc=f"Epoch {e+1}/{epochs} Validation"
        )

        # Optuna pruning hook
        if trial is not None:
            trial.report(val_acc, e)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {e+1}/{epochs} | "
              f"Train {train_acc:.4f} | Val {val_acc:.4f} | "
              f"Loss {val_loss:.4f} | Best {best_val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Checkpoint on best validation loss + early-stopping
        if val_loss < min_val_loss:
            min_val_loss     = val_loss
            best_epoch       = e + 1
            patience_counter = 0
            if trial is None and save_name is not None:
                torch.save(model.state_dict(), os.path.join(model_dir, f"{save_name}.pth"))
                print(f" ✓ Best model saved (loss {min_val_loss:.4f}) → {save_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {e+1} — best epoch {best_epoch}")
                break

    save_training_curves(save_name, train_losses, val_losses,
                         train_accuracies, val_accuracies,
                         out_dir=cfg['PATHS']['training_curves'])

    return best_val_acc

######### Training curve persistence (JSON) #########
def save_training_curves(save_name, train_losses, val_losses,
                         train_accs, val_accs, out_dir="training_curves"):
    """Save loss/accuracy history to {out_dir}/{save_name}_curves.json."""
    if save_name is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    data = {
        "train_loss":     train_losses,
        "val_loss":       val_losses,
        "train_accuracy": train_accs,
        "val_accuracy":   val_accs,
    }
    out_path = os.path.join(out_dir, f"{save_name}_curves.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Training curves saved → {out_path}")
