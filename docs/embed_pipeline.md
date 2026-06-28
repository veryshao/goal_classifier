# Embedding Pipeline

Classifies goal-setting discussions using OpenAI `text-embedding-3-large` embeddings and a scikit-learn logistic regression classifier. By default trains on `data/labels/` and evaluates on `data/eval_labels/`; both directories are overridable via CLI flags. Mirrors the BERT pipeline's train/eval split.

Embeddings are cached locally after the first run, so the OpenAI API is only called once per transcript.

---

## Prerequisites

```bash
pip install openai scikit-learn matplotlib numpy
```

### OpenAI API key

Set the key in your shell:

```bash
export OPENAI_API_KEY=sk-...
```

To persist it across sessions, add the line to `~/.zshrc`:

```bash
echo 'export OPENAI_API_KEY=sk-...' >> ~/.zshrc
source ~/.zshrc
```

Get a key at platform.openai.com → API Keys → Create new secret key. The key is never written to any file in this repository.

---

## Overview

```
--train-labels/   +  data/transcripts/
      │
      ▼
  embed.py  ──► OpenAI API (first run only, then cached)
      │               │
      │               ▼
      │       embeddings_cache/<stem>.npy
      │
      ├── build feature matrix (±2 neighbor window per utterance)
      ├── train LogisticRegression on --train-labels (default data/labels/)
      ├── save classifier → results/embed_model/classifier.joblib
      └── evaluate on --eval-labels (default data/eval_labels/) → evaluation_embed/
                                        │
                                        ▼
                              predict_embed.py  ← label new transcripts → data/labels/
                                        │
                                        └── retrain with more data (loop back to embed.py)
```

---

## Step 1 — Label transcripts

The embedding pipeline reads the same label files as the BERT pipeline. Follow the labeling instructions in [bert_pipeline.md](bert_pipeline.md) (Steps 1 through 1) to create `data/labels/<stem>.json` files. No additional labeling steps are needed.

---

## Step 2 — Run the pipeline

```bash
python embed.py
# or with alternate label directories:
python embed.py --train-labels data/binary_labels --eval-labels data/eval_binary_labels
```

On the first run, this calls the OpenAI API to embed each utterance in every transcript that has a label file. Embeddings are cached under `embeddings_cache/` so subsequent runs skip the API entirely for already-embedded transcripts.

Typical first-run output:

```
Found 24 transcripts. 24 training label file(s). 10 eval label file(s).
Loaded 3282 training utterances across 24 transcripts.
Loaded 841 eval utterances across 10 transcripts.

Computing / loading embeddings for 34 transcript(s) …
  Embedding 142 utterances for School01_Teacher01_... →
  Cached → embeddings_cache/School01_Teacher01_....npy  (shape (142, 3072))
  ...

Training on 3282 utterances, evaluating on 841 utterances …

=== Classification Report ===
              precision    recall  f1-score   support
           O       ...
           I       ...
```

---

## Output

Results are written to `evaluation_embed/`:

| File | Contents |
|------|----------|
| `classification_report.txt` | Per-class precision / recall / F1 for O and I |
| `confusion_matrix.png` | 2×2 confusion matrix |
| `per_conversation_f1.json` | Macro-F1 per eval transcript |

---

## Step 3 — Predict on new transcripts (bootstrap labeling)

Requires a trained classifier at `results/embed_model/classifier.joblib` (saved automatically by `embed.py` during Step 2).

```bash
# Review each predicted I utterance interactively
python predict_embed.py --transcript_dir data/transcripts --label_dir data/labels

# Skip review and save raw predictions
python predict_embed.py --auto --label_dir data/labels

# Single file
python predict_embed.py --file data/transcripts/<file>.txt
```

Interactive mode prints each utterance predicted as the positive class with its ±2 neighbor context and confidence scores. Accept with Enter or type a label name to override. Confirmed labels are saved to `--label_dir` in the same sparse JSON format as `data/labels/`.

If the transcript has not been embedded before, `predict_embed.py` calls the OpenAI API to embed it and caches the result under `embeddings_cache/` for future runs.

---

## Retraining loop

```
embed.py → predict_embed.py (on unlabeled sessions)
    ↑               │
    └── move confirmed labels into data/labels/ ──┘
```

Each iteration adds more labeled transcripts. Re-run `python embed.py` to retrain the classifier on the expanded label set. Already-cached embeddings are reused.

---

## How features are built

Each utterance `n` is represented by concatenating the embeddings of utterances `[n-2, n-1, n, n+1, n+2]`. Positions outside the transcript boundary are zero-padded. This gives each utterance a feature vector of size `5 × 3072 = 15 360` that encodes local conversational context without any text truncation.

---

## Caching

Each transcript's embeddings are stored as a float32 NumPy array in `embeddings_cache/<stem>.npy`. The cache directory is gitignored — embeddings are reproducible from the API and should not be committed. To force re-embedding a transcript (e.g., after the transcript file changes), delete its `.npy` file:

```bash
rm embeddings_cache/<stem>.npy
```
