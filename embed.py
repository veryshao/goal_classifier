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


def get_utterance_seconds(transcript_path: str) -> np.ndarray:
    """Return a float32 array of timestamps (seconds) for each utterance in the transcript."""
    events = parse_transcript(transcript_path)
    utterances = [e for e in events if e.event_type == "utterance"]
    return np.array([u.seconds for u in utterances], dtype=np.float32)


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
                             window: int = WINDOW,
                             timestamps: np.ndarray = None,
                             timestamp_window_seconds: int = None) -> np.ndarray:
    """
    Build per-utterance feature vectors from a (n_utterances, dim) embedding matrix.
    Output shape is always (n_utterances, (2*window+1) * dim).

    Default (timestamp_window_seconds=None): take the ±window nearest utterances by
    position, zero-pad at transcript boundaries.

    Timestamp mode (timestamps + timestamp_window_seconds provided): take up to window
    utterances before and after the target that fall within timestamp_window_seconds
    seconds, zero-pad any unfilled slots. Selection order matches positional mode —
    the window-closest neighbors are used when more than window qualify.
    """
    n, dim   = embeddings.shape
    zero_row = np.zeros(dim, dtype=np.float32)

    if timestamp_window_seconds is not None and timestamps is not None:
        features = []
        for i in range(n):
            t = timestamps[i]
            before_idx = [j for j in range(i)
                          if abs(timestamps[j] - t) <= timestamp_window_seconds]
            after_idx  = [j for j in range(i + 1, n)
                          if abs(timestamps[j] - t) <= timestamp_window_seconds]
            # Keep the `window` closest neighbors on each side
            before_embs = [embeddings[j] for j in before_idx[-window:]]
            after_embs  = [embeddings[j] for j in after_idx[:window]]
            # Zero-pad to exactly `window` slots per side
            before_pad = [zero_row] * (window - len(before_embs)) + before_embs
            after_pad  = after_embs + [zero_row] * (window - len(after_embs))
            features.append(np.concatenate(before_pad + [embeddings[i]] + after_pad))
        return np.stack(features).astype(np.float32)

    padded = np.concatenate(
        [np.tile(zero_row, (window, 1)), embeddings, np.tile(zero_row, (window, 1))],
        axis=0,
    )
    width = 2 * window + 1
    return np.stack([padded[i : i + width].ravel() for i in range(n)], axis=0)


def examples_to_Xy(examples: list[dict],
                   transcript_to_feat: dict[str, np.ndarray],
                   ) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """
    Convert a list of examples into (X, y, source_files, matched_examples) using
    pre-computed per-transcript feature matrices.
    """
    rows_X, rows_y, rows_src, rows_ex = [], [], [], []
    for ex in examples:
        feat = transcript_to_feat.get(ex["source_file"])
        if feat is None:
            continue
        idx = ex["event_idx"]
        if idx < len(feat):
            rows_X.append(feat[idx])
            rows_y.append(ex["label"])
            rows_src.append(ex["source_file"])
            rows_ex.append(ex)
    if not rows_X:
        return np.empty((0,)), np.empty((0,)), [], []
    return np.stack(rows_X), np.array(rows_y), rows_src, rows_ex


# ── Train and evaluate ────────────────────────────────────────────────────────

def train_and_evaluate(train_examples: list[dict],
                       eval_examples:  list[dict],
                       timestamp_window_seconds: int = None,
                       window: int = WINDOW,
                       output_dir: str = OUTPUT_DIR) -> None:
    """
    Train a LogisticRegression on all train_examples and evaluate on
    eval_examples, mirroring how evaluate.py works for the BERT model.
    Writes outputs to output_dir/.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all unique transcripts across both splits
    all_sources = {ex["source_file"] for ex in train_examples + eval_examples}

    print(f"Computing / loading embeddings for {len(all_sources)} transcript(s) …")
    transcript_to_feat: dict[str, np.ndarray] = {}
    for src in sorted(all_sources):
        embs = get_or_cache_embeddings(src)
        if timestamp_window_seconds is not None:
            ts = get_utterance_seconds(src)
            transcript_to_feat[src] = build_windowed_features(
                embs, window=window, timestamps=ts,
                timestamp_window_seconds=timestamp_window_seconds
            )
        else:
            transcript_to_feat[src] = build_windowed_features(embs, window=window)

    X_train, y_train, _, _              = examples_to_Xy(train_examples, transcript_to_feat)
    X_eval,  y_eval,  eval_srcs, eval_exs = examples_to_Xy(eval_examples,  transcript_to_feat)

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
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

    # ── 2. Confusion matrix ───────────────────────────────────────────────────
    cm   = confusion_matrix(y_eval, y_pred, labels=LABEL_NAMES)
    _, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix (embed)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

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

    with open(f"{output_dir}/per_conversation_f1.json", "w") as f:
        json.dump(conv_scores, f, indent=2)

    # ── 4. Error analysis ─────────────────────────────────────────────────────
    print("\n=== Error Analysis ===")
    error_groups = defaultdict(list)
    for ex, yt, yp in zip(eval_exs, y_eval.tolist(), y_pred.tolist()):
        if yt != yp:
            error_groups[(yt, yp)].append(ex)

    error_summary = {}
    for (true_lbl, pred_lbl), exs in sorted(error_groups.items()):
        key_str = f"TRUE={true_lbl} → PRED={pred_lbl}"
        print(f"\n{key_str}  ({len(exs)} errors)")
        error_summary[key_str] = []
        for ex in exs:
            target_line = next(
                (part for part in ex["text"].split("[SEP]") if "[TARGET]" in part),
                ex["text"][:120]
            )
            print(f"  [{ex['timestamp']}] {target_line.strip()[:100]}")
            print(f"    source: {ex['source_file']}")
            error_summary[key_str].append({
                "timestamp":   ex["timestamp"],
                "source_file": ex["source_file"],
                "target_text": target_line.strip(),
                "full_input":  ex["text"],
            })

    with open(f"{output_dir}/error_analysis.json", "w") as f:
        json.dump(error_summary, f, indent=2)
    print(f"\nFull error details saved to {output_dir}/error_analysis.json")

    print(f"\nAll outputs written to ./{output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding pipeline for O/I classification")
    parser.add_argument("--train-labels", default="data/labels",
                        help="Directory of training label JSON files (default: data/labels)")
    parser.add_argument("--eval-labels", default="data/eval_labels",
                        help="Directory of eval label JSON files (default: data/eval_labels)")
    parser.add_argument("--timestamp-window", type=int, default=None,
                        help="Use timestamp-based context window of N seconds instead of "
                             "the default ±utterance-count window.")
    parser.add_argument("--window", type=int, default=WINDOW,
                        help=f"Neighbor slots on each side of the target (default: {WINDOW}).")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help=f"Directory for eval outputs (default: {OUTPUT_DIR}).")
    cli_args = parser.parse_args()

    transcript_files = sorted(glob.glob("data/transcripts/*.txt"))

    train_label_files = sorted(glob.glob(f"{cli_args.train_labels}/*.json"))
    eval_label_files  = sorted(glob.glob(f"{cli_args.eval_labels}/*.json"))

    print(f"Found {len(transcript_files)} transcripts.")
    print(f"  Training labels:    {len(train_label_files)} file(s) in {cli_args.train_labels}/")
    print(f"  Evaluation labels:  {len(eval_label_files)} file(s) in {cli_args.eval_labels}/\n")

    train_examples = load_all_data(transcript_files, train_label_files,
                                   timestamp_window_seconds=cli_args.timestamp_window)
    eval_examples  = load_all_data(transcript_files, eval_label_files,
                                   timestamp_window_seconds=cli_args.timestamp_window)

    print(f"Loaded {len(train_examples)} training utterances across "
          f"{len({ex['source_file'] for ex in train_examples})} transcripts.")
    print(f"Loaded {len(eval_examples)} eval utterances across "
          f"{len({ex['source_file'] for ex in eval_examples})} transcripts.\n")

    train_and_evaluate(train_examples, eval_examples,
                       timestamp_window_seconds=cli_args.timestamp_window,
                       window=cli_args.window,
                       output_dir=cli_args.output_dir)
