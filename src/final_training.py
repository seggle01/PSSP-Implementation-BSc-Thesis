import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.train_utils import *
from src.preprocessing import reduce_q8_to_q3
from src.pssp_encoder import EnsemblePSSP, build_base_model
from src.save_predictions import *
import os, gc, joblib
from src.config import Config

######### Module-level configuration #########
cfg = Config()
device      = cfg['DEVICE']
embed_dir   = cfg['PATHS']['embed_dir']
embed_dim   = cfg['ESM2']['embed_dim']
seeds       = cfg['SEEDS']
final_runs  = cfg['FINAL_TRAINING']
rf_config   = cfg['RF_CONFIG']
window_size = cfg['WINDOW_SIZE']

for d in [cfg['PATHS']['evaluation_models'],
          cfg['PATHS']['evaluation_fasta'],
          cfg['PATHS']['evaluation_rf_dir']]:
    os.makedirs(d, exist_ok=True)

######### Full-data training (all folds combined, 80/20 split for early stopping) #########
def full_train(run_config_name, run_config):
    initial_state = run_config_name.split('_')[1]
    state         = initial_state
    if cfg['USE_8CLASS'] == run_config_name:
        state = "8class"

    fold_filenames = [
        f"{run_config['fold_prefix']}{i}_{state}.npz"
        for i in range(run_config['n_folds'])
    ]

    print(f"\n{'='*70}")
    print(f"Final Training: {run_config_name} | {run_config['n_folds']} folds")
    print(f"{'='*70}")

    # Load all folds and concatenate into a single training corpus
    loaded_folds = {}
    for name in fold_filenames:
        try:
            X, Y, ids = load_npz_dataset(os.path.join(embed_dir, name))
            if cfg['USE_8CLASS'] == run_config_name:
                Y = reduce_q8_to_q3(Y)
            loaded_folds[name] = (X, Y, ids)
            print(f"  {name}: {len(X):,} residues")
        except FileNotFoundError:
            print(f"  WARNING: {name} not found, skipping.")

    X_all = np.concatenate([loaded_folds[n][0] for n in fold_filenames if n in loaded_folds], axis=0)
    Y_all = np.concatenate([loaded_folds[n][1] for n in fold_filenames if n in loaded_folds], axis=0)
    del loaded_folds

    params              = cfg['MODEL_CONFIG'][f"{initial_state}_params"].copy()
    params['embed_dim'] = embed_dim
    epochs              = cfg['TRAINING_CONFIG'][f"{initial_state}_epochs"]

    # Train one model per seed (skip if checkpoint already exists)
    for seed in seeds:
        save_name  = f"{run_config_name}_{seed}"
        model_path = os.path.join(cfg['PATHS']['evaluation_models'], f"{save_name}.pth")

        if os.path.exists(model_path):
            print(f"  Seed {seed}: model exists, skipping → {model_path}")
            continue

        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        # Random 80/20 split (val used for early stopping)
        n     = len(X_all)
        idx   = np.arange(n)
        np.random.shuffle(idx)
        split = int(cfg['TRAINING_CONFIG']['split'] * n)

        X_train, Y_train = X_all[idx[:split]], Y_all[idx[:split]]
        X_val,   Y_val   = X_all[idx[split:]], Y_all[idx[split:]]

        model = build_base_model(params)
        optimizer, scheduler, loss_fn = build_optimizer_scheduler(model, params, epochs)

        train(
            model=model, optimizer=optimizer,
            scheduler=scheduler, loss_fn=loss_fn,
            train_data=(X_train, Y_train),
            val_data=(X_val, Y_val),
            epochs=epochs,
            model_dir=cfg['PATHS']['evaluation_models'],
            patience=run_config['patience'],
            save_name=save_name
        )

        del model, optimizer, scheduler, X_train, Y_train, X_val, Y_val
        gc.collect()
        torch.cuda.empty_cache()

    del X_all, Y_all
    gc.collect()

######### Top-level driver: four pipeline codes #########
def main():
    """
    Codes:
    1: Train final models on all folds for each config / seed.
    2: Ensemble inference on evaluation sets → save FASTA.
    3: Extract logits from training folds → train RF → save RF.
    4: RF + ER post-processing on eval sets → save FASTA.
    """
    codes = [1,2,3,4]

    for code in codes:
        # Code 1: full-data training per seed
        if code == 1:
            for run_config_name, run_config in final_runs.items():
                full_train(run_config_name, run_config)

        # Code 2: seed-ensemble inference → FASTA (base predictions)
        elif code == 2:
            for run_config_name, run_config in final_runs.items():
                initial_state = run_config_name.split('_')[1]
                state         = initial_state
                if cfg['USE_8CLASS'] == run_config_name:
                    state = "8class"

                state_dict_paths = [
                    os.path.join(cfg['PATHS']['evaluation_models'], f"{run_config_name}_{seed}.pth")
                    for seed in seeds
                ]
                missing = [p for p in state_dict_paths if not os.path.exists(p)]
                if missing:
                    print(f"  WARNING: missing models for {run_config_name}: {missing}")
                    continue

                ensemble = EnsemblePSSP(state=initial_state, state_dict_paths=state_dict_paths).to(device)

                print(f"\n{'='*70}")
                print(f"Code 2 | {run_config_name} | eval FASTA")
                print(f"{'='*70}")

                for eval_set in run_config['evaluation']:
                    eval_npz = f"{eval_set}_{state}.npz"
                    out_pred = os.path.join(cfg['PATHS']['evaluation_fasta'], f"{run_config_name}_{eval_set}_pred.fasta")
                    out_true = os.path.join(cfg['PATHS']['evaluation_fasta'], f"{run_config_name}_{eval_set}_true.fasta")

                    get_predictions_to_fasta(ensemble, eval_npz, out_pred, out_true, int(initial_state[0]), run_config_name)
                    print(f"  {eval_set}: saved → {out_pred}")

                del ensemble
                gc.collect()
                torch.cuda.empty_cache()

        # Code 3: train RF post-processor on full training corpus
        elif code == 3:
            for run_config_name, run_config in final_runs.items():
                initial_state = run_config_name.split('_')[1]
                state         = initial_state
                if cfg['USE_8CLASS'] == run_config_name:
                    state = "8class"

                rf_out = os.path.join(cfg['PATHS']['evaluation_rf_dir'], f"{run_config_name}_rf_w{window_size}.pkl")
                if os.path.exists(rf_out):
                    print(f"  {run_config_name}: RF exists, skipping → {rf_out}")
                    continue

                state_dict_paths = [
                    os.path.join(cfg['PATHS']['evaluation_models'], f"{run_config_name}_{seed}.pth")
                    for seed in seeds
                ]
                missing = [p for p in state_dict_paths if not os.path.exists(p)]
                if missing:
                    print(f"  WARNING: missing models for {run_config_name}: {missing}")
                    continue

                ensemble = EnsemblePSSP(state=initial_state, state_dict_paths=state_dict_paths).to(device)

                print(f"\n{'='*70}")
                print(f"Code 3 | {run_config_name} | training RF on full train set")
                print(f"{'='*70}")

                train_filenames = [
                    f"{run_config['fold_prefix']}{i}_{state}.npz"
                    for i in range(run_config['n_folds'])
                ]

                train_logits, train_labels, train_cids = extract_logits(
                    ensemble, train_filenames, run_config_name
                )

                del ensemble
                gc.collect()
                torch.cuda.empty_cache()

                # Build chain-safe windowed features
                X_train, y_train = build_windowed_dataset(
                    train_logits, train_labels, train_cids, window_size
                )
                print(f"  {X_train.shape[0]:,} residues | feat_dim={X_train.shape[1]}")

                # Fit Random Forest
                rf = RandomForestClassifier(**rf_config, n_jobs=-1, random_state=cfg['DEFAULT_SEED'])
                rf.fit(X_train, y_train)
                joblib.dump(rf, rf_out)
                print(f"  RF saved → {rf_out}")

                del rf, X_train, y_train, train_logits, train_labels, train_cids
                gc.collect()

        # Code 4: RF + ER post-processing on eval sets → FASTA
        elif code == 4:
            from src.preprocessing import parse_int_to_q3
            from src.external_rules import apply_all_q3_rules

            for run_config_name, run_config in final_runs.items():
                initial_state = run_config_name.split('_')[1]
                state         = initial_state
                Q = int(initial_state[0])

                rf_path = os.path.join(cfg['PATHS']['evaluation_rf_dir'], f"{run_config_name}_rf_w{window_size}.pkl")
                if not os.path.exists(rf_path):
                    print(f"  {run_config_name}: RF not found at {rf_path}, skipping.")
                    continue

                state_dict_paths = [
                    os.path.join(cfg['PATHS']['evaluation_models'], f"{run_config_name}_{seed}.pth")
                    for seed in seeds
                ]
                missing = [p for p in state_dict_paths if not os.path.exists(p)]
                if missing:
                    print(f"  WARNING: missing models for {run_config_name}: {missing}")
                    continue

                ensemble = EnsemblePSSP(state=initial_state, state_dict_paths=state_dict_paths).to(device)
                rf       = joblib.load(rf_path)

                print(f"\n{'='*70}")
                print(f"Code 4 | {run_config_name} | Q{Q} | RF + ER eval FASTA")
                print(f"{'='*70}")

                for eval_set in run_config['evaluation']:
                    eval_npz = f"{eval_set}_{state}.npz"

                    eval_logits, eval_labels, eval_cids = extract_logits(
                        ensemble, [eval_npz], run_config_name
                    )

                    # RF only
                    out_pred = os.path.join(cfg['PATHS']['evaluation_fasta'], f"{run_config_name}_{eval_set}_rf_pred.fasta")
                    out_true = os.path.join(cfg['PATHS']['evaluation_fasta'], f"{run_config_name}_{eval_set}_rf_true.fasta")
                    if not (os.path.exists(out_pred) and os.path.exists(out_true)):
                        get_rf_predictions_to_fasta(
                            rf, eval_logits, eval_labels, eval_cids,
                            window_size, out_pred, out_true, Q
                        )
                        print(f"  [{eval_set}] rf: saved → {out_pred}")

                    # ER / ER→RF / RF→ER variants (Q3 only)
                    if Q == 3:
                        # Reconstruct per-chain prob arrays and pred strings for ER variants
                        N            = len(eval_cids)
                        change_mask  = np.concatenate(([True], eval_cids[1:] != eval_cids[:-1]))
                        chain_starts = np.where(change_mask)[0]
                        chain_ends   = np.concatenate((chain_starts[1:], [N]))
                        unique_cids  = [eval_cids[s] for s in chain_starts]

                        er_pred_seqs    = []
                        er_rf_pred_seqs = []
                        rf_er_pred_seqs = []
                        true_seqs       = []

                        half = window_size // 2

                        for start, end in zip(chain_starts, chain_ends):
                            probs  = eval_logits[start:end]          # [L, Q]  already softmax
                            labels = eval_labels[start:end]          # [L]
                            L      = end - start

                            pred_str = "".join(parse_int_to_q3(int(p)) for p in probs.argmax(axis=-1))
                            true_str = "".join(parse_int_to_q3(int(l)) for l in labels)
                            true_seqs.append(true_str)

                            # Ensemble + ER
                            er_pred_seqs.append(apply_all_q3_rules(pred_str))

                            # Ensemble + ER → RF (hybrid one-hot probs)
                            Q3_MAP   = {"H": 0, "E": 1, "C": 2}
                            er_str   = apply_all_q3_rules(pred_str)
                            er_probs = probs.copy()
                            for i, (orig, ruled) in enumerate(zip(pred_str, er_str)):
                                if orig != ruled:
                                    er_probs[i]                = 0.0
                                    er_probs[i, Q3_MAP[ruled]] = 1.0

                            padded = np.zeros((L + 2 * half, Q), dtype=np.float32)
                            padded[half: half + L] = er_probs
                            shape   = (L, window_size, Q)
                            strides = (padded.strides[0], padded.strides[0], padded.strides[1])
                            X_er    = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).reshape(L, -1).copy()
                            er_rf_preds = rf.predict(X_er)
                            er_rf_pred_seqs.append("".join(parse_int_to_q3(int(p)) for p in er_rf_preds))

                            # Ensemble + RF → ER
                            padded_rf = np.zeros((L + 2 * half, Q), dtype=np.float32)
                            padded_rf[half: half + L] = probs
                            X_rf      = np.lib.stride_tricks.as_strided(padded_rf, shape=shape, strides=strides).reshape(L, -1).copy()
                            rf_preds  = rf.predict(X_rf)
                            rf_str    = "".join(parse_int_to_q3(int(p)) for p in rf_preds)
                            rf_er_pred_seqs.append(apply_all_q3_rules(rf_str))

                        # FASTA for each ER variant
                        for method, pred_seqs in [
                            ("er",    er_pred_seqs),
                            ("er_rf", er_rf_pred_seqs),
                            ("rf_er", rf_er_pred_seqs),
                        ]:
                            out_pred = os.path.join(cfg['PATHS']['evaluation_fasta'], f"{run_config_name}_{eval_set}_{method}_pred.fasta")
                            out_true = os.path.join(cfg['PATHS']['evaluation_fasta'], f"{run_config_name}_{eval_set}_{method}_true.fasta")
                         
                            with open(out_pred, "w") as fp, open(out_true, "w") as ft:
                                for cid, pred_seq, true_seq in zip(unique_cids, pred_seqs, true_seqs):
                                    fp.write(f">{cid}\n{pred_seq}\n")
                                    ft.write(f">{cid}\n{true_seq}\n")
                            print(f"  [{eval_set}] {method}: saved → {out_pred}")

                    del eval_logits, eval_labels, eval_cids
                    gc.collect()

                del ensemble, rf
                gc.collect()
                torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
