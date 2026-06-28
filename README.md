# Goal Classifier

Classifies utterances in tutoring-session transcripts as **O**utside or **I**nside a goal-setting discussion (binary O/I sequence tagging).

Two independent approaches share the same labeled data:

| Pipeline | File | Approach |
|----------|------|----------|
| **BERT** | `train.py` → `evaluate.py` | `bert-base-uncased` fine-tuned for 2-class classification |
| **Embedding** | `embed.py` | OpenAI `text-embedding-3-large` + logistic regression |

Both pipelines train on `data/labels/` and evaluate on `data/eval_labels/` by default (overridable via `--train-labels` / `--eval-labels` flags).

> **Data privacy:** All transcript and label files are excluded from version control (see `.gitignore`). Store data externally on an encrypted drive or institutional data system — never commit it to this repository.

---

## Repository layout

```
goal_classifier/
│
├── data/                        # NOT in git — study data lives here locally
│   ├── transcripts/             # Raw session transcripts (.txt)
│   ├── labels/                  # Hand-annotated labels for training (.json)
│   ├── eval_labels/             # Hold-out evaluation labels (.json)
│   ├── indexed_transcripts/     # Pre-indexed transcripts for quick lookup
│   ├── predicted_labels*/       # Output folders from predict.py runs
│   └── README.md
│
├── results/                     # NOT in git — BERT checkpoints and best_model/
├── evaluation/                  # NOT in git — BERT evaluation outputs
├── evaluation_embed/            # NOT in git — embedding pipeline evaluation outputs
├── embeddings_cache/            # NOT in git — cached OpenAI embeddings (.npy)
│
├── ── Shared utilities ──────────────────────────────────────────────────────
├── parse_transcript.py          # Transcript parser → TranscriptEvent list
├── print_indices.py             # Print utterance indices for hand-labeling
├── build_label_json.py          # One-off helper for creating label JSON files
├── label_schema.py              # Label scheme (single source of truth) and signal lists
├── prepare_data.py              # Windowed example builder + data loader
│
├── ── BERT pipeline ─────────────────────────────────────────────────────────
├── train.py                     # Fine-tune bert-base-uncased → results/best_model/
├── evaluate.py                  # Evaluate BERT model on eval_labels/
├── predict.py                   # BERT inference + interactive bootstrap labeling
│
├── ── Embedding pipeline ────────────────────────────────────────────────────
├── embed.py                     # OpenAI embeddings + logistic regression
│
└── ── Documentation ─────────────────────────────────────────────────────────
    docs/
    ├── bert_pipeline.md         # BERT pipeline walkthrough
    └── embed_pipeline.md        # Embedding pipeline walkthrough
```

---

## Setup

Install all dependencies:

```bash
pip install torch transformers scikit-learn matplotlib numpy openai
```

For the embedding pipeline only, set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Labeling transcripts (both pipelines)

Both pipelines read from `data/labels/` (training) and `data/eval_labels/` (evaluation) by default. Use `--train-labels` / `--eval-labels` to point at different directories. Labels are sparse JSON files mapping utterance index → the positive label (currently `"I"`):

```json
{ "12": "I", "13": "I", "14": "I", "27": "I" }
```

Utterance indices come from `print_indices.py`. Unlisted utterances default to the negative label (currently `"O"`).

```bash
python print_indices.py data/transcripts/<file>.txt
```

See [data/README.md](data/README.md) for the full label format.

---

## BERT pipeline

Full walkthrough: [docs/bert_pipeline.md](docs/bert_pipeline.md)

```bash
python train.py       # train on data/labels/, evaluate on data/eval_labels/
                      # saves best checkpoint → results/best_model/
python evaluate.py    # full eval report → evaluation/
python predict.py     # label new transcripts interactively

# or with alternate label directories:
python train.py    --train-labels data/binary_labels --eval-labels data/eval_binary_labels
python evaluate.py --eval-labels data/eval_binary_labels
```

---

## Embedding pipeline

Full walkthrough: [docs/embed_pipeline.md](docs/embed_pipeline.md)

```bash
python embed.py       # embed, train, evaluate → evaluation_embed/

# or with alternate label directories:
python embed.py --train-labels data/binary_labels --eval-labels data/eval_binary_labels
```

Embeddings are cached after the first run — the OpenAI API is only called once per transcript.

---

## Switching to a different label scheme

The label scheme is defined in one place: `label_schema.py`. All other scripts import from it. To switch from O/I to a different binary scheme (e.g. G/U):

1. Edit `label_schema.py`:
   ```python
   LABEL2ID = {"U": 0, "G": 1}   # was {"O": 0, "I": 1}
   DEFAULT_LABEL  = "U"           # the negative / unlisted class
   POSITIVE_LABEL = "G"           # the non-default class
   ```
   `ID2LABEL`, `LABEL_NAMES`, and `LABEL_IDS` derive automatically — no other files need editing.

2. Create label files using the new scheme (e.g. `{"12": "G", "13": "G"}`). Unlisted utterances will default to `DEFAULT_LABEL`.

3. Retrain from scratch — existing checkpoints are incompatible with a new label mapping:
   ```bash
   python train.py --train-labels data/your_labels --eval-labels data/your_eval_labels
   ```

> **Note:** The B/E legacy collapse in `load_labels_from_json` maps old B/E annotations to whatever `POSITIVE_LABEL` is set to. If you don't have legacy B/E data, this has no effect.

---

## Label format

```json
{
  "12": "I",
  "13": "I",
  "14": "I",
  "27": "I"
}
```

Sparse JSON: only positive-class utterances appear as keys. An empty `{}` means no goal discussion in that session. See [data/README.md](data/README.md) for full details.
