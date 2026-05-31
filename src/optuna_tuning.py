import torch
import torch.nn as nn
import numpy as np
import optuna
import os, gc
from pathlib import Path
from src.train_utils import *
from src.preprocessing import reduce_q8_to_q3
from src.pssp_encoder import PSSPEncoder
from src.config import Config

#########  Module-level configuration #########
cfg        = Config()
device     = cfg['DEVICE']
embed_dir  = cfg['PATHS']['embed_dir']
embed_dim  = cfg['ESM2']['embed_dim']
tuning     = cfg['TUNING']
batch_size = cfg['TRAINING_CONFIG']['batch_size']

n_trials   = cfg['TUNING_TRIALS']
tuning_dir = Path("./optuna_results")
tuning_dir.mkdir(exist_ok=True)

#########  Objective factory (loads folds once, shared across all trials) #########
def make_objective(run_config_name: str, run_config):
    tcfg       = tuning[run_config_name]
    initial_state = run_config_name.split('_')[1]                  # "3class" or "8class"
    state = initial_state
    if cfg['USE_8CLASS'] == run_config_name:
        state = "8class"
    epochs     = run_config['epochs']
    tokens     = cfg['MODEL_CONFIG'][f"{initial_state}_params"]['tokens']
    n_classes  = cfg['MODEL_CONFIG'][f"{initial_state}_params"]['num_classes']
    token_dim  = embed_dim // tokens
    use_8class = cfg['USE_8CLASS'] == run_config_name

    # Single-fold loader (shared cache across trials)
    def _load(fold_idx):
        name = f"{tcfg['fold_prefix']}{fold_idx}_{state}.npz"
        print(name)
        X, Y, _ = load_npz_dataset(os.path.join(embed_dir, name))
        if use_8class:
            Y = reduce_q8_to_q3(Y)
        return X, Y

    # Pre-load: training folds concatenated + validation fold
    print(f"\nLoading data for {run_config_name}...")
    X_train = np.concatenate([_load(i)[0] for i in tcfg['train_folds']], axis=0)
    Y_train = np.concatenate([_load(i)[1] for i in tcfg['train_folds']], axis=0)
    X_val, Y_val = _load(tcfg['val_fold'])
    print(f"  Train: {len(X_train):,} residues | Val: {len(X_val):,} residues")

    # Per-trial objective
    def objective(trial):
        set_seed(cfg['DEFAULT_SEED'])

        # Search space 
        d_ff_multiplier = trial.suggest_int('d_ff_multiplier', 4, 8)
        num_heads       = trial.suggest_categorical(
                            'num_heads',
                            [h for h in [4, 8, 16] if token_dim % h == 0]
                          )
        layers          = trial.suggest_int('layers', 6, 8)
        mlp_multiplier  = trial.suggest_int('mlp_multiplier', 4, 10)
        dropout         = trial.suggest_float('dropout', 0.1, 0.3)
        lr              = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
        weight_decay    = trial.suggest_float('weight_decay', 1e-5, 5e-3, log=True)

        # Instantiate model with the proposed configuration
        model = PSSPEncoder(
            embed_dim       = embed_dim,
            tokens          = tokens,
            d_ff_multiplier = d_ff_multiplier,
            num_heads       = num_heads,
            layers          = layers,
            mlp_multiplier  = mlp_multiplier,
            num_classes     = n_classes,
            dropout         = dropout
        ).to(device)

        trial_params = {'learning_rate': lr, 'weight_decay': weight_decay}
        optimizer, scheduler, loss_fn = build_optimizer_scheduler(model, trial_params, epochs)

        # Train with pruner-driven early stopping
        try:
            best_val_acc = train(
                model=model, optimizer=optimizer,
                scheduler=scheduler, loss_fn=loss_fn,
                train_data=(X_train, Y_train),
                val_data=(X_val, Y_val),
                epochs=epochs,
                patience=epochs,    # pruner handles early stopping, not patience
                trial=trial
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[{run_config_name}] Trial {trial.number} OOM — pruning")
            raise optuna.TrialPruned()
        finally:
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()

        return best_val_acc

    return objective

#########  Study orchestration (TPE + Median pruner, persisted to SQLite) #########
def run_study(run_config_name: str, run_config):
    tcfg         = tuning[run_config_name]
    state        = run_config_name.split('_')[1]
    epochs       = cfg['TRAINING_CONFIG'][f"{state}_epochs"]
    warmup_steps = max(5, epochs // 3)

    print(f"\n{'='*70}")
    print(f"Hyperparameter Tuning — {run_config_name}  ({epochs} epochs, {n_trials} trials)")
    print(f"{'='*70}\n")

    # Create / resume an Optuna study backed by SQLite
    study = optuna.create_study(
        storage        = f"sqlite:///{tuning_dir / tcfg['db_name']}",
        study_name     = tcfg['study_name'],
        load_if_exists = True,
        direction      = "maximize",
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = 5,
            n_warmup_steps   = warmup_steps,
            interval_steps   = 2
        ),
        sampler = optuna.samplers.TPESampler(
            n_startup_trials          = 5,
            multivariate              = True,
            warn_independent_sampling = True
        )
    )

    # Run trials
    study.optimize(
        make_objective(run_config_name, run_config),
        n_trials          = n_trials,
        show_progress_bar = True,
        gc_after_trial    = True
    )

    # Best-trial summary
    print(f"\n[{run_config_name}] Best trial : {study.best_trial.number}")
    print(f"[{run_config_name}] Best val acc: {study.best_value:.4f}")
    print(f"[{run_config_name}] Best params :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params

#########  Entry point #########
if __name__ == "__main__":
    results = {}
    for run_config_name, run_config in tuning.items():
        results[run_config_name] = run_study(run_config_name, run_config)

    print(f"\n{'='*70}")
    print("All tuning complete!")
    for name, params in results.items():
        print(f"\n{name}: {params}")
