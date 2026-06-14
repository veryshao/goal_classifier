# Goal Classifier

Classifies utterances in tutoring-session transcripts as **O**utside or **I**nside a goal-setting discussion (binary O/I sequence tagging).

Two independent approaches share the same labeled data:

| Pipeline | File | Approach |
|----------|------|----------|
| **BERT** | `train.py` → `find_best_model.py` → `evaluate.py` | `bert-base-uncased` fine-tuned with LOO cross-validation |
| **Embedding** | `embed.py` | OpenAI `text-embedding-3-large` + logistic regression |

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
├── results/                     # NOT in git — BERT fold checkpoints
├── evaluation/                  # NOT in git — BERT evaluation outputs
├── evaluation_embed/            # NOT in git — embedding pipeline evaluation outputs
├── embeddings_cache/            # NOT in git — cached OpenAI embeddings (.npy)
│
├── ── Shared utilities ──────────────────────────────────────────────────────
├── parse_transcript.py          # Transcript parser → TranscriptEvent list
├── print_indices.py             # Print utterance indices for hand-labeling
├── build_label_json.py          # One-off helper for creating label JSON files
├── label_schema.py              # O/I label definitions and signal lists
├── prepare_data.py              # Windowed example builder + data loader
│
├── ── BERT pipeline ─────────────────────────────────────────────────────────
├── train.py                     # Leave-one-out fine-tuning of bert-base-uncased
├── find_best_model.py           # Pick best LOO fold → results/best_model/
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

Both pipelines read from `data/labels/` (training) and `data/eval_labels/` (evaluation). Labels are sparse JSON files mapping utterance index → `"I"`:

```json
{ "12": "I", "13": "I", "14": "I", "27": "I" }
```

Utterance indices come from `print_indices.py`. Unlisted utterances default to `"O"`.

```bash
python print_indices.py data/transcripts/<file>.txt
```

See [data/README.md](data/README.md) for the full label format.

---

## BERT pipeline

Full walkthrough: [docs/bert_pipeline.md](docs/bert_pipeline.md)

```bash
python train.py               # LOO fine-tuning; checkpoints → results/
python find_best_model.py     # pick best fold → results/best_model/
python evaluate.py            # evaluate on eval_labels/ → evaluation/
python predict.py             # label new transcripts interactively
```

---

## Embedding pipeline

Full walkthrough: [docs/embed_pipeline.md](docs/embed_pipeline.md)

```bash
python embed.py               # embed, train, evaluate → evaluation_embed/
```

Embeddings are cached after the first run — the OpenAI API is only called once per transcript.

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

Sparse JSON: only `"I"` utterances appear as keys. An empty `{}` means no goal discussion in that session. See [data/README.md](data/README.md) for full details.
