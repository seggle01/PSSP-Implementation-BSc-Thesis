import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from train_utils import *
from preprocessing import reduce_q8_to_q3
from pssp_encoder import build_base_model, EnsemblePSSP
from save_predictions import *
import os, gc, joblib
from config import Config

######### Module-level configuration #########
cfg = Config()

device      = cfg['DEVICE']
embed_dim   = cfg['ESM2']['embed_dim']
seeds       = cfg['SEEDS']
cv_runs     = cfg['CV_RUNS']
rf_config   = cfg['RF_CONFIG']
window_size = cfg['WINDOW_SIZE']

cv_rf_dir = cfg['PATHS']['cv_rf_dir']
embed_dir = cfg['PATHS']['embed_dir']
os.makedirs(cfg['PATHS']['cv_rf_dir'], exist_ok=True)

#########  Single outer fold: nested 80/20 train/val on the held-in folds #########
def train_single_fold(fold_idx, fold_filenames, state, loaded_folds, run_config, seed):
    params              = cfg['MODEL_CONFIG'][f"{state}_params"].copy()
    params['embed_dim'] = embed_dim
    epochs              = cfg['TRAINING_CONFIG'][f"{state}_epochs"]
    set_seed(seed)

    # Held-out (outer) test fold
    test_fold_name = fold_filenames[fold_idx]

    # Build the inner train+val pool from the remaining folds
    inner_fold_names = [name for i, name in enumerate(fold_filenames) if i != fold_idx]
    X_inner, Y_inner = concat_folds(loaded_folds, inner_fold_names)
    n_inner = len(X_inner)

    # 80/20 random split inside the inner pool
    idx = np.arange(n_inner)
    np.random.shuffle(idx)
    split = int(cfg['TRAINING_CONFIG']['split'] * n_inner)
    train_idx = idx[:split]
    val_idx   = idx[split:]

    X_train, Y_train = X_inner[train_idx], Y_inner[train_idx]
    X_val,   Y_val   = X_inner[val_idx],   Y_inner[val_idx]

    # Build model / optimiser / scheduler
    model = build_base_model(params)
    optimizer, scheduler, loss_fn = build_optimizer_scheduler(model, params, epochs)

    print(f"\n{'='*70}")
    print(f"Fold {fold_idx+1}/{run_config['n_folds']} | Test (held-out): {test_fold_name}")
    print(f"{'='*70}")

    # Train on the inner train/val split
    train(
        model=model, optimizer=optimizer,
        scheduler=scheduler, loss_fn=loss_fn,
        train_data=(X_train, Y_train),
        val_data=(X_val, Y_val),
        epochs=epochs,
        model_dir=cfg['PATHS']['cv_models'],
        patience=run_config['patience'],
        save_name=f"{'_'.join(test_fold_name.split('_')[:2])}_{state}_{seed}"
    )

    # Evaluate on the held-out outer fold
    X_test, Y_test, _ = loaded_folds[test_fold_name]

    def test_loader():
        return batch_iterator(X_test, Y_test)

    test_loss, test_acc = run_phase(
        model, test_loader, loss_fn,
        optimizer=None,
        desc=f"Fold {fold_idx+1} Test"
    )

    print(f"Fold {fold_idx+1} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    del model, optimizer, scheduler, X_inner, Y_inner
    gc.collect()
    torch.cuda.empty_cache()
    return test_acc

#########  Full k-fold × multi-seed cross-validation driver #########
def run_cross_validation(run_config_name, run_config):
    # Take initial state of configuration
    initial_state = run_config_name.split('_')[1]
    state         = initial_state

    # If configuration is NetsurfP_3class use the 8class files
    if cfg['USE_8CLASS'] == run_config_name:
        state = "8class"

    fold_filenames = [
        f"{run_config['fold_prefix']}{i}_{state}.npz"
        for i in range(run_config['n_folds'])
    ]

    # Load all folds once into memory
    loaded_folds = {}
    for name in fold_filenames:
        try:
            X, Y, ids = load_npz_dataset(os.path.join(embed_dir, name))
            if cfg['USE_8CLASS'] == run_config_name:
                Y = reduce_q8_to_q3(Y)
            loaded_folds[name] = (X, Y, ids)
            print(f"Fold {name} contains {len(X)} residues")
        except FileNotFoundError:
            print(f"Fold {name} was not found. Skipping...")

    print(f"Folds loaded: {len(loaded_folds.keys())}")
    if len(loaded_folds.keys()) <= 1:
        raise Exception("Cannot continue. Need at least 2 loaded folds.")

    # Repeat across seeds (default: 42, 123, 1337)
    seed_results = {}
    for seed in seeds:
        fold_accuracies = []

        # Outer k-fold loop
        for fold_idx in range(len(loaded_folds.keys())):
            acc = train_single_fold(fold_idx, fold_filenames, initial_state, loaded_folds, run_config, seed)
            fold_accuracies.append(acc)
            print(f"  Seed {seed} | Fold {fold_idx+1} Test Acc: {acc:.4f}")

        mean_acc = sum(fold_accuracies) / run_config['n_folds']
        seed_results[seed] = {
            "fold_accuracies": fold_accuracies,
            "mean":            mean_acc,
        }
        print(f"Seed {seed} | Mean Test Acc over {run_config['n_folds']} folds: {mean_acc:.4f}")

    # Aggregate summary
    print(f"\n{'='*70}")
    print(f"Cross-Validation Summary — {run_config_name}")
    print(f"{'='*70}")

    all_means = []
    for seed, result in seed_results.items():
        print(f"\nSeed {seed}:")
        for fold_i, acc in enumerate(result['fold_accuracies']):
            print(f"  Fold {fold_i+1:>2} Test Acc: {acc:.4f}")
        print(f"=> Mean: {result['mean']:.4f}")
        all_means.append(result['mean'])

    overall_mean = sum(all_means) / len(all_means)
    print(f"\n{'─'*70}")
    print(f"Overall Mean Acc across all seeds: {overall_mean:.4f}")
    print(f"{'='*70}\n")

    del loaded_folds

#########  Top-level driver: three pipeline codes #########
def main():
    """
    Codes:
    1: Do k-fold Cross-Validation for each configuration for 3 seeds.
    2: Get fasta of cross-validation models by averaging across seeds.
    3: Do k-fold Cross-Validation for each configuration using Random Forest
       post-processing and save the fasta predictions.
    """
    codes = [1,2,3]

    for code in codes:
        # Code 1: pure base-model k-fold CV
        if code == 1:
            for run_config_name, run_config in cv_runs.items():
                run_cross_validation(run_config_name, run_config)

        # Code 2: seed-ensemble inference → FASTA 
        elif code == 2:
            for run_config_name, run_config in cv_runs.items():
                print(f"\nGenerating FASTA: {run_config_name} | {run_config['n_folds']}")

                initial_state = run_config_name.split('_')[1]
                state         = initial_state
                if cfg['USE_8CLASS'] == run_config_name:
                    state = "8class"

                fold_filenames = [f"{run_config['fold_prefix']}{i}_{state}.npz"         for i in range(run_config['n_folds'])]
                fold_prefixes  = [f"{run_config['fold_prefix']}{i}_{initial_state}"     for i in range(run_config['n_folds'])]

                for filename, prefix in zip(fold_filenames, fold_prefixes):
                    # Collect per-seed checkpoints
                    state_dict_paths = [os.path.join(cfg['PATHS']['cv_models'], f"{prefix}_{seed}.pth") for seed in seeds]

                    # Skip if any seed model is missing
                    missing = [p for p in state_dict_paths if not os.path.exists(p)]
                    if missing:
                        print(f"  WARNING: Missing model files for {filename}: {missing}")
                        continue

                    ensemble = EnsemblePSSP(state=initial_state, state_dict_paths=state_dict_paths).to(device)

                    out_pred = os.path.join(cfg['PATHS']['cv_fasta'], f"{prefix}_pred.fasta")
                    out_true = os.path.join(cfg['PATHS']['cv_fasta'], f"{prefix}_true.fasta")

                    get_predictions_to_fasta(ensemble, filename, out_pred, out_true, int(initial_state[0]), run_config_name)

        # Code 3: train RF post-processor per fold + FASTA
        elif code == 3:
            for run_config_name, run_config in cv_runs.items():
                initial_state = run_config_name.split('_')[1]
                state         = initial_state
                if cfg['USE_8CLASS'] == run_config_name:
                    state = "8class"

                n_folds        = run_config['n_folds']
                fold_filenames = [f"{run_config['fold_prefix']}{i}_{state}.npz"         for i in range(n_folds)]
                fold_prefixes  = [f"{run_config['fold_prefix']}{i}_{initial_state}"     for i in range(n_folds)]

                for fold_idx in range(n_folds):
                    prefix          = fold_prefixes[fold_idx]
                    test_filename   = fold_filenames[fold_idx]
                    train_filenames = [fold_filenames[j] for j in range(n_folds) if j != fold_idx]

                    state_dict_paths = [
                        os.path.join(cfg['PATHS']['cv_models'], f"{prefix}_{seed}.pth")
                        for seed in seeds
                    ]

                    ensemble = EnsemblePSSP(state=initial_state, state_dict_paths=state_dict_paths).to(device)

                    # Extract ensemble probabilities on train + test 
                    train_logits, train_labels, train_cids = extract_logits(ensemble, train_filenames, run_config_name)
                    test_logits,  test_labels,  test_cids  = extract_logits(ensemble, [test_filename], run_config_name)

                    # Build chain-safe windowed feature matrices
                    X_train, y_train = build_windowed_dataset(train_logits, train_labels, train_cids, window_size)
                    X_test,  y_test  = build_windowed_dataset(test_logits,  test_labels,  test_cids,  window_size)

                    print(f"Fold {fold_idx}: train={X_train.shape[0]:,} | test={X_test.shape[0]:,}")

                    # Train Random Forest post-processor
                    rf = RandomForestClassifier(**rf_config, n_jobs=-1, random_state=cfg['DEFAULT_SEED'])
                    rf.fit(X_train, y_train)

                    acc = (rf.predict(X_test) == y_test).mean()
                    print(f"  Fold {fold_idx}: RF Test Acc = {acc:.4f}")

                    # RF + per-chain FASTA outputs
                    rf_out = os.path.join(cv_rf_dir, f"{prefix}_rf_w{window_size}.pkl")
                    joblib.dump(rf, rf_out)
                    print(f"  Fold {fold_idx}: RF saved → {rf_out}")

                    out_pred = os.path.join(cfg['PATHS']['cv_fasta'], f"{prefix}_rf_pred.fasta")
                    out_true = os.path.join(cfg['PATHS']['cv_fasta'], f"{prefix}_rf_true.fasta")
                    get_rf_predictions_to_fasta(rf, test_logits, test_labels, test_cids, window_size, out_pred, out_true, int(initial_state[0]))

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
