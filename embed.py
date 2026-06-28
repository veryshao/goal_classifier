"""
OpenAI embedding pipeline for goal-discussion classification.

Each transcript's utterances are embedded once and cached to
  embeddings_cache/<transcript_stem>.npy
as a float32 array of shape (n_utterances, EMBEDDING_DIM).

Feature vectors concatenate embeddings of the ±WINDOW neighbors around
each target utterance (zero-padded at boundaries), giving a feature of
size (2*WINDOW+1) × EMBEDDING_DIM.

Training and evaluation mirror the BERT pipeline:
  - Trains on all transcripts with labels in data/labels/
  - Evaluates on all transcripts with labels in data/eval_labels/
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import joblib
import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from parse_transcript import parse_transcript
from prepare_data import load_all_data
from label_schema import LABEL_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
CACHE_DIR       = "embeddings_cache"
OUTPUT_DIR      = "evaluation_embed"
MODEL_SAVE_DIR  = "results/embed_model"
WINDOW          = 2      # ±2 neighbors → 5 embeddings concatenated per example
EMBED_BATCH     = 100    # texts per API call (API max is 2048)
MAX_RETRIES     = 3
RETRY_DELAY     = 2.0    # seconds between retries

client = OpenAI()


# ── Embedding helpers ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings via the OpenAI API in batches of EMBED_BATCH.
    Returns float32 array of shape (len(texts), EMBEDDING_DIM).
    """
    all_vecs = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        for attempt in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                vecs = [item.embedding
                        for item in sorted(response.data, key=lambda x: x.index)]
                all_vecs.extend(vecs)
                break
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Embedding error (attempt {attempt + 1}): {exc}"
                          f" — retrying in {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    return np.array(all_vecs, dtype=np.float32)


def get_or_cache_embeddings(transcript_path: str) -> np.ndarray:
    """
    Return cached embeddings for a transcript, or compute and cache them.
    Embeds each utterance as "<speaker>: <text>".
    Shape: (n_utterances, EMBEDDING_DIM).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem       = os.path.splitext(os.path.basename(transcript_path))[0]
    cache_path = os.path.join(CACHE_DIR, f"{stem}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    events     = parse_transcript(transcript_path)
    utterances = [e for e in events if e.event_type == "utterance"]

    if not utterances:
        empty = np.zeros((0, 3072), dtype=np.float32)
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
    Output shape: (n_utterances, (2*window+1) * embedding_dim).
    """
    n, dim   = embeddings.shape
    zero_row = np.zeros(dim, dtype=np.float32)
    padded   = np.concatenate(
        [np.tile(zero_row, (window, 1)), embeddings, np.tile(zero_row, (window, 1))],
        axis=0,
    )
    width = 2 * window + 1
    return np.stack([padded[i : i + width].ravel() for i in range(n)], axis=0)


def examples_to_Xy(examples: list[dict],
                   transcript_to_feat: dict[str, np.ndarray],
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert a list of examples into (X, y, source_files) arrays using
    pre-computed per-transcript feature matrices.
    """
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
    """
    Train a LogisticRegression on all train_examples and evaluate on
    eval_examples, mirroring how evaluate.py works for the BERT model.
    Writes outputs to OUTPUT_DIR/.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all unique transcripts across both splits
    all_sources = {ex["source_file"] for ex in train_examples + eval_examples}

    print(f"Computing / loading embeddings for {len(all_sources)} transcript(s) …")
    transcript_to_feat: dict[str, np.ndarray] = {}
    for src in sorted(all_sources):
        embs = get_or_cache_embeddings(src)
        transcript_to_feat[src] = build_windowed_features(embs)

    X_train, y_train, _          = examples_to_Xy(train_examples, transcript_to_feat)
    X_eval,  y_eval,  eval_srcs  = examples_to_Xy(eval_examples,  transcript_to_feat)

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

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, "classifier.joblib")
    joblib.dump(clf, save_path)
    print(f"Saved classifier → {save_path}")

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
    ax.set_title("Confusion Matrix (embed)")
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
    parser = argparse.ArgumentParser(description="Embedding pipeline for O/I classification")
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
