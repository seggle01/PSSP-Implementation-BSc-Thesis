import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.preprocessing import preprocess_tgt_Q3, preprocess_tgt_Q8
from src.config import Config

######### Module-level configuration #########
cfg       = Config()
device    = cfg['DEVICE']
data_dir  = cfg['PATHS']['data_dir']
embed_dir = cfg['PATHS']['embed_dir']

######### ESM-2 Protein Language Model initialisation #########
def init_ESM2(model_name):
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading ESM-2 model ({model_name}).")

    # Instantiate tokenizer + frozen encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = AutoModel.from_pretrained(model_name).to(device)

    # Freeze parameters and set eval mode
    for param in esm_model.parameters():
        param.requires_grad = False
    esm_model.eval()

    return tokenizer, esm_model

######### Plain-text dataset parser (FASTA-like: name / seq / target triplets) #########
def create_dataset_as_df(file_path: str):
    assert isinstance(file_path, str),     "file_path must be str."
    assert os.path.exists(file_path),      f"File path doesn't exist: {file_path}"

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    rows     = []
    pid_count = {}                         
    seq_name = seq = tgt = expecting = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith('>'):
            raw_pid = line.replace(' ', '').replace('>', '') 

            # Duplicate-PID handling
            if raw_pid in pid_count:                          
                pid_count[raw_pid] += 1                        
                seq_name = f"{raw_pid}-{pid_count[raw_pid]}"  
                print(f"  [duplicate PID] '{raw_pid}' → '{seq_name}'")
            else:                                             
                pid_count[raw_pid] = 0                         
                seq_name = raw_pid                             

            expecting = 'seq'

        elif expecting == 'seq':
            seq       = line.replace(' ', '')
            expecting = 'target'
        elif expecting == 'target':
            tgt       = line.replace(' ', '')
            rows.append({'name': seq_name, 'seq': seq, 'tgt': tgt})
            expecting = None

    n_total  = len(rows)
    n_unique = len(pid_count)
    print(f"Dataset loaded: {n_total} segments ({n_unique} unique PIDs, {n_total - n_unique} duplicates renamed)")
    return pd.DataFrame(rows)

######### ESM-2 embedding extraction → compressed NPZ archive #########
def extract_embeddings_to_npz(dataset: pd.DataFrame, out_filename: str, state: int):
    # Load PLM
    tokenizer, model = init_ESM2(model_name=cfg['ESM2']['model_name'])

    all_embeddings, all_labels, all_chain_ids = [], [], []

    # Per-sample loop 
    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        chain_id  = dataset['name'].iloc[i]
        sequence  = dataset['seq'].iloc[i]
        target    = dataset['tgt'].iloc[i]

        # Map DSSP characters to integer labels
        if state == 3:
            labels = preprocess_tgt_Q3(target)
        elif state == 8:
            labels = preprocess_tgt_Q8(target)

        # Forward through ESM-2, dropping <BOS>/<EOS> rows
        with torch.no_grad():
            encoded    = tokenizer(sequence, return_tensors='pt').to(device)
            output     = model(**encoded)
            embeddings = output.last_hidden_state[0, 1:-1, :]  # [L, 1280]

        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        labels_np     = np.array(labels, dtype=np.int8)

        # Sanity: per-residue alignment
        assert len(embeddings_np) == len(labels_np), \
            f"Embedding/label length mismatch for {chain_id}: {len(embeddings_np)} vs {len(labels_np)}"

        all_embeddings.append(embeddings_np)
        all_labels.append(labels_np)
        all_chain_ids.append(np.array([chain_id] * len(labels_np)))

        del encoded, output, embeddings
        torch.cuda.empty_cache()

    # Flatten across all proteins to a residue-level record 
    embeddings_arr = np.concatenate(all_embeddings, axis=0)   # [N, 1280]
    labels_arr     = np.concatenate(all_labels,     axis=0)   # [N]
    chain_ids_arr  = np.concatenate(all_chain_ids,  axis=0)   # [N]

    print(f"Total residues: {len(embeddings_arr)}")

    output_path = out_filename + ".npz"
    np.savez_compressed(output_path,
                        embeddings = embeddings_arr,
                        labels     = labels_arr,
                        chain_ids  = chain_ids_arr)

    print(f"Saved: {output_path}")

######### Driver #########
def main():
    # List of datasets to extract PLM embeddings for
    datasets = [
        # Examples
        "PISCES_3class/CASP13.txt",
        "NetsurfP_3class/CB513.txt",
        "NetsurfP_8class/CB513.txt"
    ]

    for file_path in datasets:
        # Infer the state (3 or 8) from path affix
        state = None
        if '8class' in file_path:
            state = 8
        elif '3class' in file_path:
            state = 3
        else:
            raise Exception("Make sure filepath contains affix '_3class' or '_8class'.")

        # Parse text file into dataframe
        dataset = create_dataset_as_df(os.path.join(data_dir, file_path))

        filename    = file_path.split('/')[-1]
        base_name   = filename.split('.')[0]
        output_path = os.path.join(embed_dir, f"{base_name}_{state}class")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Embed + save
        extract_embeddings_to_npz(
            dataset      = dataset,
            out_filename = output_path,
            state        = state
        )

        del dataset
        torch.cuda.empty_cache()

    print("All datasets processed successfully.")

if __name__ == "__main__":
    main()
