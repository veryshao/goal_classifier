"""
BERT embedding pipeline for goal-discussion classification.

Uses bert-base-uncased as a frozen feature extractor: each utterance is
embedded individually as "<speaker>: <text>" (mean-pooled last hidden state,
768-dim), cached to bert_embeddings_cache/<stem>.npy, then windowed features
are built by concatenating embeddings for the ±WINDOW neighbors around each
target utterance (zero-padded at boundaries) → 5 × 768 = 3 840 dims.

This is the embed-first, concatenate-second approach, mirroring embed.py's
architecture but using BERT's contextual representations in place of the
OpenAI API. Compare with train.py, which concatenates utterance text first
and lets BERT attend across utterance boundaries end-to-end.

Because BERT weights are frozen, the cache is stable across runs.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from parse_transcript import parse_transcript
from prepare_data import load_all_data
from label_schema import LABEL_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
CACHE_DIR  = "bert_embeddings_cache"
OUTPUT_DIR = "evaluation_bert_embed"
WINDOW     = 2       # ±2 neighbors → 5 embeddings concatenated per example
BATCH_SIZE = 32      # utterances per forward pass
MAX_LENGTH = 128     # tokens per utterance (no window text, so 128 suffices)
EMBED_DIM  = 768     # bert-base hidden size

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert      = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _mean_pool(last_hidden_state: torch.Tensor,
               attention_mask:    torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings via frozen bert-base-uncased in batches.
    Returns float32 array of shape (len(texts), EMBED_DIM).
    """
    all_vecs = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            enc   = tokenizer(
                batch,
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            out  = bert(**enc)
            vecs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            all_vecs.append(vecs.cpu().float().numpy())
    return np.concatenate(all_vecs, axis=0)


def get_or_cache_embeddings(transcript_path: str) -> np.ndarray:
    """
    Return cached BERT embeddings for a transcript, or compute and cache them.
    Each utterance is embedded as "<speaker>: <text>".
    Shape: (n_utterances, EMBED_DIM).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem       = os.path.splitext(os.path.basename(transcript_path))[0]
    cache_path = os.path.join(CACHE_DIR, f"{stem}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    events     = parse_transcript(transcript_path)
    utterances = [e for e in events if e.event_type == "utterance"]

    if not utterances:
        empty = np.zeros((0, EMBED_DIM), dtype=np.float32)
        np.save(cache_path, empty)
        return empty

    texts = [f"{u.speaker}: {u.text}" for u in utterances]
    print(f"  Embedding {len(texts)} utterances for {stem} …")
    embeddings = embed_texts(texts)
    np.save(cache_path, embeddings)
    print(f"  Cached → {cache_path}  (shape {embeddings.shape})")
    return embeddings


# ── Feature construction ──────────────────────────────────────────────────────

def build_windowed_features(embeddings: np.ndarray,
                             window: int = WINDOW) -> np.ndarray:
    """
    For each utterance i, concatenate embeddings[i-window … i+window].
    Out-of-bounds positions are zero-padded.
    Output shape: (n_utterances, (2*window+1) * EMBED_DIM).
    """
    n, dim   = embeddings.shape
    zero_row = np.zeros(dim, dtype=np.float32)
    padded   = np.concatenate(
        [np.tile(zero_row, (window, 1)), embeddings, np.tile(zero_row, (window, 1))],
        axis=0,
    )
    width = 2 * window + 1
    return np.stack([padded[i : i + width].ravel() for i in range(n)], axis=0)


def examples_to_Xy(
    examples:             list[dict],
    transcript_to_feat:   dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows_X, rows_y, rows_src = [], [], []
    for ex in examples:
        feat = transcript_to_feat.get(ex["source_file"])
        if feat is None:
            continue
        idx = ex["event_idx"]
        if idx < len(feat):
            rows_X.append(feat[idx])
            rows_y.append(ex["label"])
            rows_src.append(ex["source_file"])
    if not rows_X:
        return np.empty((0,)), np.empty((0,)), []
    return np.stack(rows_X), np.array(rows_y), rows_src


# ── Train and evaluate ────────────────────────────────────────────────────────

def train_and_evaluate(train_examples: list[dict],
                       eval_examples:  list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_sources = {ex["source_file"] for ex in train_examples + eval_examples}

    print(f"Computing / loading BERT embeddings for {len(all_sources)} transcript(s) …")
    transcript_to_feat: dict[str, np.ndarray] = {}
    for src in sorted(all_sources):
        embs = get_or_cache_embeddings(src)
        transcript_to_feat[src] = build_windowed_features(embs)

    X_train, y_train, _         = examples_to_Xy(train_examples, transcript_to_feat)
    X_eval,  y_eval,  eval_srcs = examples_to_Xy(eval_examples,  transcript_to_feat)

    if len(X_train) == 0:
        print("No training examples — check your --train-labels directory.")
        return
    if len(X_eval) == 0:
        print("No evaluation examples — check your --eval-labels directory.")
        return

    print(f"\nTraining on {len(X_train)} utterances, "
          f"evaluating on {len(X_eval)} utterances …")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)

    # ── 1. Classification report ──────────────────────────────────────────────
    print("\n=== Classification Report ===")
    report = classification_report(
        y_eval, y_pred,
        target_names=LABEL_NAMES,
        labels=LABEL_NAMES,
        zero_division=0,
    )
    print(report)
    with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
        f.write(report)

    # ── 2. Confusion matrix ───────────────────────────────────────────────────
    cm   = confusion_matrix(y_eval, y_pred, labels=LABEL_NAMES)
    _, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix (bert-embed)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

    # ── 3. Per-conversation F1 ────────────────────────────────────────────────
    print("\n=== Per-Conversation F1 (macro) ===")
    per_conv: dict[str, dict] = defaultdict(lambda: {"true": [], "pred": []})
    for src, yt, yp in zip(eval_srcs, y_eval.tolist(), y_pred.tolist()):
        per_conv[src]["true"].append(yt)
        per_conv[src]["pred"].append(yp)

    conv_scores: dict[str, float] = {}
    for conv, data in sorted(per_conv.items()):
        f1 = f1_score(data["true"], data["pred"],
                      average="macro", zero_division=0)
        conv_scores[conv] = round(f1, 3)
        print(f"  {os.path.basename(conv):50s}  macro-F1 = {f1:.3f}")

    with open(f"{OUTPUT_DIR}/per_conversation_f1.json", "w") as f:
        json.dump(conv_scores, f, indent=2)

    print(f"\nAll outputs written to ./{OUTPUT_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT embedding pipeline for O/I classification")
    parser.add_argument("--train-labels", default="data/labels",
                        help="Directory of training label JSON files (default: data/labels)")
    parser.add_argument("--eval-labels", default="data/eval_labels",
                        help="Directory of eval label JSON files (default: data/eval_labels)")
    cli_args = parser.parse_args()

    transcript_files = sorted(glob.glob("data/transcripts/*.txt"))

    train_label_files = sorted(glob.glob(f"{cli_args.train_labels}/*.json"))
    eval_label_files  = sorted(glob.glob(f"{cli_args.eval_labels}/*.json"))

    print(f"Found {len(transcript_files)} transcripts.")
    print(f"  Training labels:    {len(train_label_files)} file(s) in {cli_args.train_labels}/")
    print(f"  Evaluation labels:  {len(eval_label_files)} file(s) in {cli_args.eval_labels}/\n")

    train_examples = load_all_data(transcript_files, train_label_files)
    eval_examples  = load_all_data(transcript_files, eval_label_files)

    print(f"Loaded {len(train_examples)} training utterances across "
          f"{len({ex['source_file'] for ex in train_examples})} transcripts.")
    print(f"Loaded {len(eval_examples)} eval utterances across "
          f"{len({ex['source_file'] for ex in eval_examples})} transcripts.\n")

    train_and_evaluate(train_examples, eval_examples)
