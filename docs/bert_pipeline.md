# BERT Pipeline

Fine-tunes `bert-base-uncased` for binary O/I classification of goal-setting discussions in tutoring transcripts.

---

## Prerequisites

```bash
pip install torch transformers scikit-learn matplotlib numpy
```

GPU (CUDA or Apple MPS) is recommended for training. Inference in `evaluate.py` and `predict.py` auto-selects the best available device.

---

## Overview

```
label transcripts
       │
       ▼
  python train.py          ← LOO fine-tuning; one checkpoint per fold
       │
       ▼
  python find_best_model.py ← picks best fold → results/best_model/
       │
       ├──► python evaluate.py   ← eval on data/eval_labels/ → evaluation/
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

## Step 2 — Train (leave-one-out cross-validation)

```bash
python train.py
```

Holds out one transcript at a time, trains on the rest, evaluates on the held-out session, and saves a checkpoint under `results/<transcript_stem>/`. Class weights are recomputed per fold from the training split only — the held-out session's label distribution never leaks into its own fold's loss.

Training prints a per-fold O/I classification report at each epoch.

---

## Step 3 — Pick the best model

```bash
python find_best_model.py
```

Reads `trainer_state.json` from every fold checkpoint, filters to folds whose held-out session has at least one `I` label, picks the fold with the highest macro-F1, and copies that checkpoint to `results/best_model/`. Always run this after `train.py` and before `evaluate.py` or `predict.py`.

Output looks like:

```
0.812  School01_Teacher01_...txt  (checkpoint-45)
0.743  School01_Teacher07_...txt  (checkpoint-40)  (no I labels in held-out fold — excluded)
...
mean macro-F1 across N folds: 0.789

Best fold with I-label support: School01_Teacher01_...txt  macro-F1=0.812
Copied results/School01_.../checkpoint-45 -> results/best_model
```

---

## Step 4 — Evaluate

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

## Step 5 — Predict on new transcripts (bootstrap labeling)

```bash
# Review each predicted I utterance interactively
python predict.py --transcript_dir data/transcripts --label_dir data/predicted_labels_new

# Skip review and save raw predictions
python predict.py --auto --label_dir data/predicted_labels_new

# Single file
python predict.py --file data/transcripts/<file>.txt --label_dir data/predicted_labels_new
```

Interactive mode (`default`) prints each predicted `I` utterance with its context window and confidence scores. Accept with Enter or type `I`/`O` to override.

Confirmed labels are saved to `--label_dir` in the same sparse JSON format as `data/labels/`. Move confirmed files into `data/labels/` and retrain to grow the labeled set.

---

## Retraining loop

```
train.py → find_best_model.py → predict.py (on unlabeled sessions)
       ↑                                         │
       └─────── move confirmed labels ───────────┘
                into data/labels/
```

Each iteration adds more labeled transcripts, improving the model's precision on the most common error types visible in `error_analysis.json`.
