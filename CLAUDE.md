# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project that classifies utterances in tutoring-session transcripts as either **Outside (O)**
or **Inside (I)** a goal-setting discussion. It's a per-utterance binary sequence-tagging task with
two approaches:

1. **BERT classifier** — `bert-base-uncased` fine-tuned for 2-class O/I sequence classification
   (`transformers` + `torch`).
2. **Embedding classifier** — OpenAI `text-embedding-3-large` embeddings with ±2 neighbor windowed
   features, trained with a `LogisticRegression` LOO classifier (`openai` + `scikit-learn`).

There is no build system, package manifest, or test suite — this is a small data-science script
collection run directly with `python <script>.py`. Dependencies: `torch`, `transformers`,
`scikit-learn`, `matplotlib`, `numpy`, `openai`.

## Commands

- Inspect how a transcript parses: `python parse_transcript.py data/transcripts/<file>.txt`
- List utterances with their indices (needed for hand-labeling): `python print_indices.py data/transcripts/<file>.txt`
- Inspect data loading / windowed examples end-to-end: `python prepare_data.py`
- Train (leave-one-conversation-out): `python train.py` — fine-tunes `bert-base-uncased` once per
  held-out transcript, writing per-fold checkpoints to `./results/<conversation_name>/`.
- Pick and install best model checkpoint: `python find_best_model.py` — ranks all LOO folds by
  macro-F1, picks the best fold with at least one `I` label in its held-out conversation, and
  copies that checkpoint to `./results/best_model/`. Run this after `train.py` and before
  `evaluate.py` or `predict.py`.
- Run evaluation: `python evaluate.py` — expects a fine-tuned model at `./results/best_model`,
  writes classification report, confusion matrix, error analysis, and per-conversation F1 to `./evaluation/`.
- Predict + bootstrap-label new sessions: `python predict.py [--file <path>] [--auto]` — runs the
  saved `./results/best_model` over transcripts, interactively reviews flagged `I` predictions
  (unless `--auto` is passed), and writes confirmed labels to `data/labels/`.
- Compute and cache OpenAI embeddings, run LOO logistic regression: `python embed.py`

There is no lint/test command. To exercise an individual piece, run the relevant module directly.

Detailed pipeline walkthroughs live in `docs/`: `bert_pipeline.md` and `embed_pipeline.md`.

## Pipeline / architecture

Data flows through these stages:

1. **`data/transcripts/*.txt`** — raw session transcripts. Each file starts with a metadata header
   line `(school=..., teacher=..., session_date=..., session_time=..., tutor=...)` followed by
   timestamped content:
   - utterances: `[mm:ss] Tutor|Student (Name): text`
   - interaction/app events: `[mm:ss] [app_switch|mouse click|keyboard type|... event: details]`
   Filenames encode `School##_Teacher##_<date>_<time>_Stu_<id>`.

2. **`parse_transcript.py`** — regex-based parser (`UTTERANCE_RE`, `APP_EVENT_RE`) that turns a
   transcript file into a flat, timestamp-sorted list of `TranscriptEvent` dataclass instances.
   Timestamps may be `mm:ss` or `h:mm:ss`. Note: transcripts can interleave multiple student/tutor
   pairs, so sorting by `seconds` is only an approximation of true chronological order.

3. **`print_indices.py`** — prints `[index] [timestamp] [speaker] text` for every utterance in
   timestamp-sorted order (via `parse_transcript`). The printed index is the key used in label JSON
   files — this script is the human labeling aid.

4. **`data/labels/*.json`** — sparse O/I annotations: a JSON object mapping an utterance **index**
   (string, 0-based among utterances only) to `"I"`. An empty object `{}` means the session has no
   goal-discussion; unlisted utterances default to `"O"`. Legacy `"B"`/`"E"` values from an older
   4-class scheme are silently mapped to `"I"` by `load_labels_from_json`.

5. **`build_label_json.py`** — a one-off templated helper for creating label files from
   `print_indices.py` output. The output filename is hardcoded (`session1_labels.json`) — hand-edit
   it per session.

6. **`label_schema.py`** — defines `LABEL2ID`/`ID2LABEL` for the 2-class O/I scheme (`O=0, I=1`),
   annotation guidance for what counts as `I` vs `O`, and two heuristic signal lists:
   `GOAL_APP_SIGNALS` and `GOAL_VERBAL_SIGNALS`.

7. **`prepare_data.py`** — turns parsed events + sparse labels into model-ready examples:
   - `get_app_context_around` collects non-utterance events within ±30s and renders them as tags
     like `[SCREEN: ...]` / `[CLICK: ...]` / `[DRAG: ...]` / etc.
   - `make_windowed_examples` builds, per utterance, a `[SEP]`-joined window of ±2 surrounding
     utterances with the target wrapped in `[TARGET] ...`, plus an `[APP_CTX] ...` suffix.
     Utterances are indexed by position among utterances only (`event_idx`), matching
     `print_indices.py` output and the keys in label JSON files.
   - `load_labels_from_json` reads the sparse label dict; maps legacy B/E → I.
   - `load_all_data(transcript_files, label_files)` matches files by filename stem, skips unmatched
     transcripts, fills unlabeled utterances with `"O"`, and returns dicts with `text`, `label`,
     `label_id`, `source_file`, `timestamp`, and `event_idx` keys.

8. **`train.py`** — fine-tunes `bert-base-uncased` (`AutoModelForSequenceClassification`, 2 labels)
   for the O/I task. Module-level: config constants, `tokenizer` (`MAX_LENGTH = 384`), `GoalDataset`,
   and `WeightedTrainer` (weighted cross-entropy; class weights recomputed per-fold from training
   split only). Data loading and LOO training live inside `if __name__ == "__main__":` — keep it
   that way, because `evaluate.py` imports `MAX_LENGTH` from this module and must not trigger a
   training run as a side effect.

9. **`find_best_model.py`** — post-training utility that ranks all LOO fold checkpoints under
   `./results/*/` by their `best_metric` (macro-F1) from `trainer_state.json`, filters to folds
   whose held-out conversation has at least one `I` label, picks the top-scoring eligible fold, and
   copies its best checkpoint to `./results/best_model/` (overwriting any prior contents). Run this
   after `train.py` completes and before `evaluate.py` or `predict.py`.

11. **`evaluate.py`** — loads all transcripts+labels via `load_all_data` (from `data/eval_labels/`),
   runs batched inference (batch size 32, CUDA→MPS→CPU device placement) over the saved model from
   `./results/best_model`, and writes to `./evaluation/`:
   - `classification_report.txt` (O/I precision/recall/F1)
   - `confusion_matrix.png`
   - `error_analysis.json` (misclassified examples grouped by `(true, pred)`)
   - `per_conversation_f1.json` (macro-F1 per source transcript)

10. **`predict.py`** — inference + bootstrap-labeling tool:
    - `predict_file` runs the model over every utterance in a transcript (utterance-only index space,
      0-based) and returns predictions with confidence and per-class probabilities.
    - `review_predictions` is an interactive reviewer: prints each utterance predicted `I` with its
      context window and lets you accept or override from the keyboard (`I`/`O`).
    - `save_label_json` writes confirmed labels to `data/labels/<transcript_stem>.json`.
    - `get_unlabeled_files` finds transcripts without a matching label file.
    - CLI: `--file <path>` for a single transcript, `--auto` to skip review.
    - Uses CUDA→MPS→CPU device placement; imports `MAX_LENGTH` from `train`.

12. **`embed.py`** — OpenAI embedding pipeline + logistic regression classifier, mirroring the BERT
    pipeline's train/eval split:
    - Embeds each utterance as `"<speaker>: <text>"` via `text-embedding-3-large`.
    - Caches per-transcript embeddings as `embeddings_cache/<stem>.npy` (float32, shape
      `(n_utterances, 3072)`). Cache files are gitignored and reproducible from the API.
    - Builds windowed feature vectors by concatenating embeddings for indices `[n-2, n-1, n, n+1,
      n+2]` with zero-padding at boundaries → feature dim = 15 360.
    - Trains `LogisticRegression(class_weight="balanced")` on all `data/labels/` examples, then
      evaluates on all `data/eval_labels/` examples. Writes classification report, confusion matrix,
      and per-conversation F1 to `evaluation_embed/`.

## Working in this codebase

- **Label scheme is O/I only** — `LABEL2ID = {"O": 0, "I": 1}`. All B/E annotations in the data
  files were converted; `load_labels_from_json` maps any remaining legacy B/E → I defensively. Do
  not reintroduce B/E without updating `label_schema.py`, `train.py`, `evaluate.py`, `predict.py`,
  and `embed.py` consistently.
- **Utterance index space** — indices are 0-based positions among utterances only, not among all
  events. `print_indices.py`, `make_windowed_examples`, `predict_file`, and `load_labels_from_json`
  all use this convention. Never index by position in the full `events` list.
- The example `text` format (`[SEP]`-joined context with `[TARGET]` and `[APP_CTX]`) is defined in
  `prepare_data.make_windowed_examples` and consumed as-is by `train.py`, `evaluate.py`, and
  `predict.py` — keep them in sync if you change it.
- `evaluate.py` and `predict.py` expect a single consolidated checkpoint at `./results/best_model`.
  `train.py`'s LOO loop writes one checkpoint per fold under `./results/<conversation_name>/` —
  pick the best fold and copy it to `./results/best_model` before running eval or prediction.
- `train.py`'s data loading and LOO training loop live inside `if __name__ == "__main__":` — keep
  it that way. `evaluate.py` imports `MAX_LENGTH` from `train`; any code at `train` import time
  would fire as a side effect.
- `evaluate.py` and `predict.py` run their model-loading/inference logic at module level. That's
  fine because nothing imports either of them — but guard that code before adding such an import.
- All data files (`data/`, `results/`, `evaluation/`, `embeddings_cache/`) are gitignored. Never
  commit transcripts, labels, embeddings, or model weights.
