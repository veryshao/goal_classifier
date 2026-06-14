# BERT Pipeline

Fine-tunes `bert-base-uncased` for binary O/I classification of goal-setting discussions in tutoring transcripts. Trains on `data/labels/` and evaluates on `data/eval_labels/`, saving the best checkpoint (by macro-F1 on the eval set) to `results/best_model/`.

---

## Prerequisites

```bash
pip install torch transformers scikit-learn matplotlib numpy
```

GPU (CUDA or Apple MPS) is strongly recommended for training. Inference in `evaluate.py` and `predict.py` auto-selects the best available device.

---

## Overview

```
label transcripts
       │
       ▼
  python train.py      ← trains on data/labels/, evaluates on data/eval_labels/,
       │                  saves best checkpoint → results/best_model/
       │
       ├──► python evaluate.py   ← full eval report on data/eval_labels/ → evaluation/
       │
       └──► python predict.py    ← label new transcripts → data/labels/
                  │
                  └── retrain with more data (loop back to train.py)
```

---

## Step 1 — Label transcripts

Print utterance indices for a transcript (these are the keys in label JSON files):

```bash
python print_indices.py data/transcripts/<file>.txt
```

Create `data/labels/<same-stem>.json` with `"I"` for every utterance inside a goal discussion:

```json
{
  "12": "I",
  "13": "I",
  "14": "I",
  "27": "I"
}
```

Utterances not listed default to `"O"`. An empty `{}` means no goal discussion in that session.

Verify the parser and data loader see what you expect:

```bash
python parse_transcript.py data/transcripts/<file>.txt
python prepare_data.py
```

`prepare_data.py` prints the O/I label distribution and a sample windowed example across all matched transcript/label pairs.

---

## Step 2 — Train

```bash
python train.py
```

Trains `bert-base-uncased` on all transcripts in `data/labels/`, evaluating on `data/eval_labels/` after each epoch. At the end it saves the epoch with the highest eval macro-F1 to `results/best_model/` automatically — no manual checkpoint selection needed.

Training prints an O/I classification report after every eval epoch.

To keep training running while your laptop is closed (on a remote machine):

```bash
nohup python train.py > results/train_log.txt 2>&1 &
tail -f results/train_log.txt   # check progress
```

---

## Step 3 — Evaluate

```bash
python evaluate.py
```

Runs batched inference over all transcripts in `data/eval_labels/` using the model at `results/best_model/`. Writes to `evaluation/`:

| File | Contents |
|------|----------|
| `classification_report.txt` | Per-class precision / recall / F1 for O and I |
| `confusion_matrix.png` | 2×2 confusion matrix |
| `error_analysis.json` | Misclassified examples grouped by (true, pred) |
| `per_conversation_f1.json` | Macro-F1 per eval transcript |

---

## Step 4 — Predict on new transcripts (bootstrap labeling)

```bash
# Review each predicted I utterance interactively
python predict.py --transcript_dir data/transcripts --label_dir data/predicted_labels_new

# Skip review and save raw predictions
python predict.py --auto --label_dir data/predicted_labels_new

# Single file
python predict.py --file data/transcripts/<file>.txt --label_dir data/predicted_labels_new
```

Interactive mode prints each predicted `I` utterance with its context window and confidence scores. Accept with Enter or type `I`/`O` to override. Confirmed labels are saved to `--label_dir` in the same sparse JSON format as `data/labels/`.

---

## Retraining loop

```
train.py → predict.py (on unlabeled sessions)
    ↑               │
    └── move confirmed labels into data/labels/ ──┘
```

Each iteration adds more labeled transcripts. `error_analysis.json` from `evaluate.py` shows which error types are most common and where to focus labeling effort.
