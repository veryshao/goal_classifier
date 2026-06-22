# BERT Embedding Pipeline

Classifies goal-setting discussions using `bert-base-uncased` as a frozen feature extractor and a scikit-learn logistic regression classifier. Trains on `data/labels/` and evaluates on `data/eval_labels/`, mirroring the other two pipelines' train/eval split.

BERT embeddings are cached locally after the first run, so the model only performs inference once per transcript.

---

## How this differs from the other two pipelines

| | BERT fine-tuning (`train.py`) | OpenAI embeddings (`embed.py`) | BERT embeddings (`bert_embed.py`) |
|---|---|---|---|
| **Encoder** | `bert-base-uncased`, fine-tuned | `text-embedding-3-large`, frozen | `bert-base-uncased`, frozen |
| **Context handling** | Utterances concatenated as text first, BERT attends across them | Each utterance embedded separately, neighbor vectors concatenated | Each utterance embedded separately, neighbor vectors concatenated |
| **Feature dim** | Sequence → logit (end-to-end) | 5 × 3 072 = 15 360 | 5 × 768 = 3 840 |
| **Classifier** | Fine-tuned classification head | LogisticRegression | LogisticRegression |
| **Requires API key** | No | Yes | No |
| **Caching** | Model weights only | `embeddings_cache/<stem>.npy` | `bert_embeddings_cache/<stem>.npy` |

Because BERT weights are frozen, the cache is stable: cached embeddings never go stale.

---

## Prerequisites

```bash
pip install torch transformers scikit-learn matplotlib numpy
```

No API key is required. The `bert-base-uncased` weights are downloaded from Hugging Face on the first run and cached locally by the `transformers` library.

GPU (CUDA or Apple MPS) speeds up the embedding pass but is not required — the script auto-selects the best available device.

---

## Overview

```
data/labels/  +  data/transcripts/
      │
      ▼
bert_embed.py  ──► bert-base-uncased (first run only, then cached)
      │                     │
      │                     ▼
      │         bert_embeddings_cache/<stem>.npy
      │
      ├── build feature matrix (±2 neighbor window per utterance)
      ├── train LogisticRegression on data/labels/
      └── evaluate on data/eval_labels/ → evaluation_bert_embed/
```

---

## Step 1 — Label transcripts

This pipeline reads the same label files as the other two pipelines. Follow the labeling instructions in [bert_pipeline.md](bert_pipeline.md) (Step 1) to create `data/labels/<stem>.json` files. No additional steps are needed.

---

## Step 2 — Run the pipeline

```bash
python bert_embed.py
```

On the first run, `bert-base-uncased` embeds every utterance in every transcript that has a label file. Embeddings are cached under `bert_embeddings_cache/` so subsequent runs skip inference entirely for already-embedded transcripts.

Typical first-run output:

```
Using device: mps
Found 24 transcripts.
  Training labels:    24 file(s) in data/labels/
  Evaluation labels:  10 file(s) in data/eval_labels/

Loaded 3282 training utterances across 24 transcripts.
Loaded 841 eval utterances across 10 transcripts.

Computing / loading BERT embeddings for 34 transcript(s) …
  Embedding 142 utterances for School01_Teacher01_... …
  Cached → bert_embeddings_cache/School01_Teacher01_....npy  (shape (142, 768))
  ...

Training on 3282 utterances, evaluating on 841 utterances …

=== Classification Report ===
              precision    recall  f1-score   support
           O       ...
           I       ...
```

---

## Output

Results are written to `evaluation_bert_embed/`:

| File | Contents |
|------|----------|
| `classification_report.txt` | Per-class precision / recall / F1 for O and I |
| `confusion_matrix.png` | 2×2 confusion matrix |
| `per_conversation_f1.json` | Macro-F1 per eval transcript |

---

## How features are built

Each utterance is embedded individually as `"<speaker>: <text>"` using mean pooling over BERT's last hidden state (768-dim). Utterance `n` is then represented by concatenating the embeddings of utterances `[n-2, n-1, n, n+1, n+2]`, zero-padded at transcript boundaries. This gives a feature vector of `5 × 768 = 3 840` dims that encodes local conversational context.

Unlike `train.py`, self-attention does not cross utterance boundaries — each utterance's embedding is computed independently before the window is assembled.

---

## Caching

Each transcript's embeddings are stored as a float32 NumPy array in `bert_embeddings_cache/<stem>.npy`. The cache directory is gitignored. To force re-embedding a transcript (e.g., after its text changes), delete its cache file:

```bash
rm bert_embeddings_cache/<stem>.npy
```
