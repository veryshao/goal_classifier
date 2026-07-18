# BERT Pipeline

Fine-tunes `bert-base-uncased` for binary O/I classification of goal-setting discussions in tutoring transcripts. By default trains on `data/labels/` and evaluates on `data/eval_labels/`; both directories are overridable via CLI flags. Saves the best checkpoint (by macro-F1 on the eval set) to `results/best_model/`.

> **Annotation schemes:** two hand-annotation sets exist in this repo — **O/I** (Outside/Inside a goal discussion; `data/labels/`, `data/eval_labels/`) and **U/R** (Unrelated/Related to goal discussion; `data/binary_labels/`, `data/eval_binary_labels/`). This pipeline trains on whichever scheme is active in `label_schema.py`; point the label-directory flags at the matching directories. See `label_schema.py` and CLAUDE.md for how to switch schemes.

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
  python train.py      ← trains on --train-labels (default data/labels/),
       │                  evaluates on --eval-labels (default data/eval_labels/),
       │                  saves best checkpoint → results/best_model/
       │
       ├──► python evaluate.py   ← eval report on --eval-labels → evaluation/
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
# or with alternate label directories:
python train.py --train-labels data/binary_labels --eval-labels data/eval_binary_labels
```

Trains `bert-base-uncased` on all transcripts matched by the training labels directory (default `data/labels/`), evaluating on the eval labels directory (default `data/eval_labels/`) after each epoch. At the end it saves the epoch with the highest eval macro-F1 to `results/best_model/` automatically — no manual checkpoint selection needed.

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
# or with an alternate eval labels directory:
python evaluate.py --eval-labels data/eval_binary_labels
```

Runs batched inference over all transcripts matched by the eval labels directory (default `data/eval_labels/`) using the model at `results/best_model/`. Writes to `evaluation/`:

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

## Configuring the context window

The context window determines which surrounding utterances are concatenated (as `[SEP]`-separated text) around each `[TARGET]` utterance before BERT processes it. Two modes are available.

### Utterance-count window (default)

By default, the ±2 nearest utterances by position are included. To change the count, edit the `utterance_window` argument in the `make_windowed_examples` call inside `load_all_data` in `prepare_data.py`:

```python
# prepare_data.py — load_all_data
examples = make_windowed_examples(events, full_labels, utterance_window=3)  # ±3 utterances
```

### Timestamp-based window

Pass `--timestamp-window N` to use all utterances within N seconds of the target instead:

```bash
python train.py --timestamp-window 30
```

When `--timestamp-window` is set, the utterance-count window is ignored. Note that goal conversations tend to span 1–3 minutes, so a 30-second window typically captures 5–15 utterances depending on conversation pace.

**Any window change requires retraining from scratch** — the input distribution changes and the existing checkpoint at `results/best_model/` becomes invalid.

---

## Retraining loop

```
train.py → predict.py (on unlabeled sessions)
    ↑               │
    └── move confirmed labels into data/labels/ ──┘
```

Each iteration adds more labeled transcripts. `error_analysis.json` from `evaluate.py` shows which error types are most common and where to focus labeling effort.
