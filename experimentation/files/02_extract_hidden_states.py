import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer

SEED = 42
torch.manual_seed(SEED)

DATA_DIR = Path("data")
STIMULI_PATH = DATA_DIR / "stimuli.json"
MODEL_NAME = "gpt2"
MAX_SEQ_LEN = 1024
DEVICE = "cpu" # ça marche pas avec mps a cause des hidden state nan



def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded. n_layers={model.config.n_layer}, "
          f"hidden_size={model.config.n_embd}")
    return tokenizer, model



@torch.no_grad()
def get_all_hidden_states(text: str, tokenizer, model):
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_SEQ_LEN,
        truncation=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)

    out = model(input_ids)
    hidden_states = [h.squeeze(0).cpu().float().numpy()
                     for h in out.hidden_states]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return hidden_states, tokens



def last_token_per_layer(hidden_states):
    return np.stack([h[-1] for h in hidden_states])



def mean_tokens_per_layer(hidden_states):
    return np.stack([h.mean(axis=0) for h in hidden_states])



def incremental_snapshots(hidden_states):
    seq_len = hidden_states[0].shape[0]
    snapshots = []
    for t in range(1, seq_len + 1):
        snap = np.stack([h[:t] for h in hidden_states])
        snapshots.append(snap)
    return snapshots



def word_to_token_alignment(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    alignment = []
    word_boundaries = []
    current_word_start = None

    for i, tok in enumerate(tokens):
        if tok.startswith("Ġ") or i == 0:
            if current_word_start is not None:
                word_boundaries.append((current_word_start, i - 1))
            current_word_start = i
    if current_word_start is not None:
        word_boundaries.append((current_word_start, len(tokens) - 1))

    return word_boundaries


def main():
    with open(STIMULI_PATH, encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} sentence pairs.")

    tokenizer, model = load_model()
    n_layers_plus1 = model.config.n_layer + 1
    hidden_size = model.config.n_embd

    N = len(pairs)
    layerwise = np.zeros((N * 2, n_layers_plus1, hidden_size), dtype=np.float32)
    layerwise_mean = np.zeros((N * 2, n_layers_plus1, hidden_size), dtype=np.float32)

    N_WORD_BINS = 10
    incremental_vectors = [
        [[[] for _ in range(N_WORD_BINS)] for _ in range(n_layers_plus1)]
        for _ in range(2)
    ]

    token_counts = []

    print("Extracting hidden states...")
    for i, pair in enumerate(tqdm(pairs)):
        for cond_idx, text_key in enumerate(["semantic", "jabberwocky"]):
            text = pair[text_key]

            hs, tokens = get_all_hidden_states(text, tokenizer, model)
            word_boundaries = word_to_token_alignment(text, tokens)
            seq_len = hs[0].shape[0]

            layerwise[cond_idx * N + i] = last_token_per_layer(hs)
            layerwise_mean[cond_idx * N + i] = mean_tokens_per_layer(hs)

            n_words = len(word_boundaries)
            for bin_idx in range(N_WORD_BINS):
                w_idx = int(round(bin_idx * (n_words - 1) / (N_WORD_BINS - 1)))
                tok_start, tok_end = word_boundaries[w_idx]
                rep_tok = tok_end
                for layer in range(n_layers_plus1):
                    vec = hs[layer][rep_tok]
                    incremental_vectors[cond_idx][layer][bin_idx].append(vec)

            if cond_idx == 0:
                token_counts.append({
                    "id": pair["id"],
                    "semantic_ntok": seq_len,
                })
            else:
                token_counts[i]["jabberwocky_ntok"] = seq_len
                token_counts[i]["bpe_ratio"] = round(
                    seq_len / token_counts[i]["semantic_ntok"], 3
                )

    np.savez_compressed(
        DATA_DIR / "hidden_states_layerwise.npz",
        data=layerwise,
        condition=np.array([0] * N + [1] * N, dtype=np.int8),
    )
    print(f"Saved layerwise hidden states → {DATA_DIR}/hidden_states_layerwise.npz")

    np.savez_compressed(
        DATA_DIR / "hidden_states_layerwise_mean.npz",
        data=layerwise_mean,
        condition=np.array([0] * N + [1] * N, dtype=np.int8),
    )
    print(f"Saved layerwise mean hidden states → {DATA_DIR}/hidden_states_layerwise_mean.npz")

    inc_arrays = {}
    for cond in range(2):
        cond_name = "semantic" if cond == 0 else "jabberwocky"
        for layer in range(n_layers_plus1):
            for bin_idx in range(N_WORD_BINS):
                vecs = incremental_vectors[cond][layer][bin_idx]
                if vecs:
                    key = f"{cond_name}_L{layer:02d}_W{bin_idx:02d}"
                    inc_arrays[key] = np.stack(vecs)

    np.savez_compressed(DATA_DIR / "hidden_states_incremental.npz", **inc_arrays)
    print(f"Saved incremental hidden states → {DATA_DIR}/hidden_states_incremental.npz")

    with open(DATA_DIR / "token_counts.json", "w") as f:
        json.dump(token_counts, f, indent=2)

    avg_ratio = sum(t["bpe_ratio"] for t in token_counts) / len(token_counts)
    print(f"\nBPE token ratio (jabber/semantic): mean={avg_ratio:.3f}")


if __name__ == "__main__":
    main()