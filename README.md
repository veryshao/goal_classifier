# Goal Classifier

Classifies utterances in tutoring-session transcripts as **O**utside or **I**nside a
goal-setting discussion (binary sequence tagging). Two parallel annotation schemes are
supported; the active scheme is configured in `label_schema.py`.

Three independent ML approaches share the same labeled data, plus a statistics script
that requires no model:

| Pipeline | Files | Approach |
|----------|-------|----------|
| **BERT fine-tuning** | `train.py` → `evaluate.py` | `bert-base-uncased` fine-tuned end-to-end |
| **OpenAI embedding + LR** | `embed.py` | `text-embedding-3-large` embeddings + logistic regression |
| **BERT embedding + LR** | `bert_embed.py` | Frozen `bert-base-uncased` embeddings + logistic regression |
| **Goal statistics** | `goal_stats.py` | Descriptive stats from hand labels (no model required) |

All pipelines train on `data/labels/` and evaluate on `data/eval_labels/` by default
(overridable via `--train-labels` / `--eval-labels`).

> **Data privacy:** All transcript and label files are excluded from version control
> (see `.gitignore`). Store data externally on an encrypted drive or institutional data
> system — never commit it to this repository.

---

## Repository layout

```
goal_classifier/
│
├── data/                         # NOT in git — study data lives here locally
│   ├── transcripts/              # Raw session transcripts (.txt)
│   ├── labels/                   # Hand-annotated O/I training labels (.json)
│   ├── eval_labels/              # Hold-out O/I evaluation labels (.json)
│   ├── binary_labels/            # Hand-annotated U/R training labels (.json)
│   ├── eval_binary_labels/       # Hold-out U/R evaluation labels (.json)
│   ├── predicted_labels*/        # Output folders from predict*.py runs
│   └── README.md
│
├── results/                      # NOT in git — BERT checkpoints → best_model/
├── evaluation/                   # NOT in git — BERT pipeline evaluation outputs
├── evaluation_embed/             # NOT in git — OpenAI embedding eval outputs
├── evaluation_bert_embed/        # NOT in git — BERT embedding eval outputs
├── evaluation_goal_stats/        # NOT in git — goal_stats.py outputs
├── embeddings_cache/             # NOT in git — cached OpenAI embeddings (.npy)
├── bert_embeddings_cache/        # NOT in git — cached BERT embeddings (.npy)
│
├── ── Shared utilities ──────────────────────────────────────────────────────────
├── parse_transcript.py           # Transcript parser → TranscriptEvent list
├── print_indices.py              # Print utterance indices for hand-labeling
├── build_label_json.py           # One-off helper for creating label JSON files
├── label_schema.py               # Active label scheme + both ANNOTATION_SCHEMES
├── prepare_data.py               # Windowed example builder + data loader
│
├── ── BERT fine-tuning pipeline ─────────────────────────────────────────────────
├── train.py                      # Fine-tune bert-base-uncased → results/best_model/
├── evaluate.py                   # Evaluate BERT model on eval_labels/
├── predict.py                    # BERT inference + interactive bootstrap labeling
│
├── ── OpenAI embedding pipeline ─────────────────────────────────────────────────
├── embed.py                      # OpenAI text-embedding-3-large + logistic regression
├── predict_embed.py              # Embedding model inference + bootstrap labeling
│
├── ── BERT embedding pipeline ───────────────────────────────────────────────────
├── bert_embed.py                 # Frozen bert-base-uncased embeddings + logistic regression
├── predict_bert_embed.py         # BERT embedding model inference + bootstrap labeling
│
├── ── Goal statistics ───────────────────────────────────────────────────────────
├── goal_stats.py                 # Descriptive stats from hand labels (no model needed)
│
└── ── Documentation ─────────────────────────────────────────────────────────────
    docs/
    ├── bert_pipeline.md          # BERT fine-tuning pipeline walkthrough
    ├── embed_pipeline.md         # OpenAI embedding pipeline walkthrough
    ├── bert_embed_pipeline.md    # BERT embedding pipeline walkthrough
    └── goal_stats.md             # Goal statistics script usage and interpretation
```

---

## Setup

Install all dependencies:

```bash
pip install torch transformers scikit-learn matplotlib numpy openai
```

For the OpenAI embedding pipeline only, set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Annotation schemes

Two parallel hand-annotation sets cover the same sessions:

| Scheme | Labels | Meaning | Label directories |
|--------|--------|---------|-------------------|
| **O/I** | `O` / `I` | **I**nside a goal-setting discussion (span-style: includes transitions and asides within the episode) | `data/labels/`, `data/eval_labels/` |
| **U/R** | `U` / `R` | **R**elated to goal discussion (per-utterance: only utterances that are themselves goal talk) | `data/binary_labels/`, `data/eval_binary_labels/` |

The active scheme for the training/eval/predict pipelines is set in `label_schema.py`.
`goal_stats.py` processes both schemes independently via `ANNOTATION_SCHEMES`. See
`CLAUDE.md` for how to switch the active scheme.

---

## Labeling transcripts

Labels are sparse JSON files mapping utterance index → the positive label:

```json
{ "12": "I", "13": "I", "14": "I", "27": "I" }
```

Utterance indices come from `print_indices.py`. Unlisted utterances default to the
negative label. An empty `{}` means no goal discussion.

```bash
python print_indices.py data/transcripts/<file>.txt
```

See `data/README.md` for the full label format.

---

## BERT fine-tuning pipeline

Full walkthrough: [docs/bert_pipeline.md](docs/bert_pipeline.md)

```bash
python train.py       # train on data/labels/, evaluate on data/eval_labels/
python evaluate.py    # full eval report → evaluation/
python predict.py     # label new transcripts interactively

# U/R scheme:
python train.py    --train-labels data/binary_labels --eval-labels data/eval_binary_labels
python evaluate.py --eval-labels data/eval_binary_labels
```

---

## OpenAI embedding pipeline

Full walkthrough: [docs/embed_pipeline.md](docs/embed_pipeline.md)

```bash
python embed.py          # embed + train + evaluate → evaluation_embed/
python predict_embed.py  # label new transcripts interactively
```

Embeddings are cached in `embeddings_cache/` after the first run.

---

## BERT embedding pipeline

Full walkthrough: [docs/bert_embed_pipeline.md](docs/bert_embed_pipeline.md)

```bash
python bert_embed.py          # embed + train + evaluate → evaluation_bert_embed/
python predict_bert_embed.py  # label new transcripts interactively
```

Embeddings are cached in `bert_embeddings_cache/`. No API key required.

---

## Goal statistics

Full walkthrough: [docs/goal_stats.md](docs/goal_stats.md)

```bash
python goal_stats.py               # both schemes → evaluation_goal_stats/OI/ and .../UR/
python goal_stats.py --schemes OI  # one scheme only
```

No model or API key required — reads hand labels directly.

---

## Switching the active label scheme

The active scheme is defined in `label_schema.py` (currently U/R). All training/eval/predict
pipelines import from it. To switch:

1. Edit `label_schema.py`:
   ```python
   LABEL2ID = {"O": 0, "I": 1}   # was {"U": 0, "R": 1}
   DEFAULT_LABEL  = "O"
   POSITIVE_LABEL = "I"
   ```
   `ID2LABEL`, `LABEL_NAMES`, and `LABEL_IDS` derive automatically.

2. Point the pipeline at the matching label directories:
   ```bash
   python train.py --train-labels data/labels --eval-labels data/eval_labels
   ```

3. Retrain from scratch — existing checkpoints are incompatible with a new label mapping.

`ANNOTATION_SCHEMES` in `label_schema.py` documents both sets and their directories for
reference. See `CLAUDE.md` for the full switch procedure.
