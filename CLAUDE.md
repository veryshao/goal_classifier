# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project that classifies utterances in tutoring-session transcripts as either inside or
outside a goal-setting discussion. It's a per-utterance binary sequence-tagging task. Two annotation
schemes are in use across different label directories (O/I and U/R — see `label_schema.py`); the
active scheme is set there. Three pipeline approaches:

1. **BERT classifier** — `bert-base-uncased` fine-tuned for 2-class U/R sequence classification
   (`transformers` + `torch`). Trains on `data/labels/`, evaluates on `data/eval_labels/`.
2. **OpenAI embedding classifier** — OpenAI `text-embedding-3-large` embeddings with ±2 neighbor
   windowed features and a `LogisticRegression` classifier (`openai` + `scikit-learn`). Same
   train/eval split.
3. **BERT embedding classifier** — frozen `bert-base-uncased` mean-pooled embeddings with ±2
   neighbor windowed features and a `LogisticRegression` classifier (`torch` + `transformers` +
   `scikit-learn`). Same train/eval split; no API key required.

There is no build system, package manifest, or test suite — this is a small data-science script
collection run directly with `python <script>.py`. Dependencies: `torch`, `transformers`,
`scikit-learn`, `matplotlib`, `numpy`, `openai`.

## Commands

- Inspect how a transcript parses: `python parse_transcript.py data/transcripts/<file>.txt`
- List utterances with their indices (needed for hand-labeling): `python print_indices.py data/transcripts/<file>.txt`
- Inspect data loading / windowed examples end-to-end: `python prepare_data.py`
- Train BERT: `python train.py [--train-labels DIR] [--eval-labels DIR] [--timestamp-window N]` —
  fine-tunes `bert-base-uncased` on the training labels directory (default `data/labels/`),
  evaluates on the eval labels directory (default `data/eval_labels/`) each epoch, and saves the
  best checkpoint to `./results/best_model/`. `--timestamp-window N` switches context windowing
  from the default ±2 utterance-count window to a ±N-second timestamp window.
- Run BERT evaluation: `python evaluate.py [--eval-labels DIR]` — expects a model at
  `./results/best_model`, evaluates on the given label directory (default `data/eval_labels/`),
  writes classification report, confusion matrix, error analysis, and per-conversation F1 to
  `./evaluation/`.
- Predict + bootstrap-label new sessions: `python predict.py [--file <path>] [--auto]` — runs the
  saved `./results/best_model` over transcripts, interactively reviews flagged positive-class
  predictions (unless `--auto` is passed), and writes confirmed labels to `data/labels/`.
- Run embedding pipeline: `python embed.py [--train-labels DIR] [--eval-labels DIR] [--timestamp-window N]` —
  embeds utterances via OpenAI API (cached), trains logistic regression on the training labels
  directory (default `data/labels/`), evaluates on the eval labels directory (default
  `data/eval_labels/`), saves classifier to `./results/embed_model/`, writes eval results to
  `evaluation_embed/`. `--timestamp-window N` switches neighbor selection from index-based to
  time-based (up to `WINDOW` neighbors within N seconds, zero-padded); feature dim unchanged.
- Predict with embedding pipeline: `python predict_embed.py [--file <path>] [--auto] [--timestamp-window N]` —
  runs the saved `./results/embed_model/classifier.joblib` over transcripts, interactively reviews
  flagged predictions (unless `--auto` is passed), and writes confirmed labels to `data/labels/`.
  Pass `--timestamp-window N` if the classifier was trained with that flag.
- Run BERT embedding pipeline: `python bert_embed.py [--train-labels DIR] [--eval-labels DIR] [--timestamp-window N]` —
  embeds utterances via frozen `bert-base-uncased` (cached), trains logistic regression, saves
  classifier to `./results/bert_embed_model/`, writes eval results to `evaluation_bert_embed/`.
  `--timestamp-window N` switches neighbor selection to time-based; feature dim unchanged.
- Predict with BERT embedding pipeline: `python predict_bert_embed.py [--file <path>] [--auto] [--timestamp-window N]` —
  same as `predict_embed.py` but for the BERT embedding classifier. Pass `--timestamp-window N` if
  the classifier was trained with that flag.

There is no lint/test command. To exercise an individual piece, run the relevant module directly.

Detailed pipeline walkthroughs live in `docs/`: `bert_pipeline.md`, `embed_pipeline.md`, and
`bert_embed_pipeline.md`.

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

4. **`data/labels/*.json`** — sparse annotations for training: a JSON object mapping an utterance
   **index** (string, 0-based among utterances only) to the positive label defined in
   `label_schema.py` (e.g. `"I"` or `"R"` depending on the active scheme). An empty object `{}`
   means the session has no goal-discussion; unlisted utterances default to `DEFAULT_LABEL`.
   Legacy `"B"`/`"E"` values from an older 4-class scheme are silently mapped to `POSITIVE_LABEL`
   by `load_labels_from_json`.

5. **`data/eval_labels/*.json`** — same format as `data/labels/`, but held out for evaluation only.
   Never used during training.

6. **`build_label_json.py`** — a one-off templated helper for creating label files from
   `print_indices.py` output. The output filename is hardcoded (`session1_labels.json`) — hand-edit
   it per session.

7. **`label_schema.py`** — single source of truth for the label scheme. Defines `LABEL2ID`/`ID2LABEL`
   for the 2-class scheme (currently `U=0, R=1`), plus derived constants used by all other scripts:
   `DEFAULT_LABEL` (the negative/unlisted class, `"U"`), `POSITIVE_LABEL` (the non-default class,
   `"R"`), `LABEL_NAMES` (ordered by ID for reports), and `LABEL_IDS`. Also contains annotation
   guidance for what counts as `R` vs `U`, and two heuristic signal lists: `GOAL_APP_SIGNALS` and
   `GOAL_VERBAL_SIGNALS`.

8. **`prepare_data.py`** — turns parsed events + sparse labels into model-ready examples:
   - `get_app_context_around` collects non-utterance events within ±30s and renders them as tags
     like `[SCREEN: ...]` / `[CLICK: ...]` / `[DRAG: ...]` / etc.
   - `make_windowed_examples` builds, per utterance, a `[SEP]`-joined window of surrounding
     utterances with the target wrapped in `[TARGET] ...`, plus an `[APP_CTX] ...` suffix.
     Accepts two mutually exclusive windowing modes via keyword arguments:
     - `utterance_window=2` (default) — includes the ±N nearest utterances by position
     - `timestamp_window_seconds=N` — includes all utterances within N seconds of the target's
       timestamp; when set, `utterance_window` is ignored
     Utterances are indexed by position among utterances only (`event_idx`), matching
     `print_indices.py` output and the keys in label JSON files.
   - `load_labels_from_json` reads the sparse label dict; maps legacy B/E → `POSITIVE_LABEL`.
   - `load_all_data(transcript_files, label_files, timestamp_window_seconds=None)` matches files
     by filename stem, skips unmatched transcripts, fills unlabeled utterances with `DEFAULT_LABEL`,
     and returns dicts with `text`, `label`, `label_id`, `source_file`, `timestamp`, and
     `event_idx` keys. `timestamp_window_seconds` is forwarded to `make_windowed_examples`.

9. **`train.py`** — fine-tunes `bert-base-uncased` (`AutoModelForSequenceClassification`, 2 labels)
   for the U/R task. Module-level: config constants, `tokenizer` (`MAX_LENGTH = 384`), `GoalDataset`,
   and `WeightedTrainer` (weighted cross-entropy using class weights computed from the training split).
   The `__main__` block accepts `--train-labels` and `--eval-labels` flags (defaulting to
   `data/labels/` and `data/eval_labels/`), trains for `EPOCHS` epochs evaluating after each one,
   and saves the epoch with the highest eval macro-F1 to `./results/best_model/` via
   `trainer.save_model()`. Keep all data loading and training inside `if __name__ == "__main__":` —
   `evaluate.py` and `predict.py` both import `MAX_LENGTH` from this module and must not trigger a
   training run as a side effect.

10. **`evaluate.py`** — accepts an `--eval-labels` flag (default `data/eval_labels/`), loads all
    transcripts+labels via `load_all_data`, runs batched inference (batch size 32, CUDA→MPS→CPU
    device placement) over the saved model from `./results/best_model`, and writes to `./evaluation/`:
    - `classification_report.txt` (U/R precision/recall/F1)
    - `confusion_matrix.png`
    - `error_analysis.json` (misclassified examples grouped by `(true, pred)`)
    - `per_conversation_f1.json` (macro-F1 per source transcript)

11. **`predict.py`** — inference + bootstrap-labeling tool:
    - `predict_file` runs the model over every utterance in a transcript (utterance-only index space,
      0-based) and returns predictions with confidence and per-class probabilities.
    - `review_predictions` is an interactive reviewer: prints each utterance predicted `R` with its
      context window and lets you accept or override from the keyboard (`U`/`R`).
    - `save_label_json` writes confirmed labels to `data/labels/<transcript_stem>.json`.
    - `get_unlabeled_files` finds transcripts without a matching label file.
    - CLI: `--file <path>` for a single transcript, `--auto` to skip review.
    - Uses CUDA→MPS→CPU device placement; imports `MAX_LENGTH` from `train`.

12. **`embed.py`** — OpenAI embedding pipeline + logistic regression classifier, mirroring the BERT
    pipeline's train/eval split. Accepts `--train-labels` and `--eval-labels` flags (defaulting to
    `data/labels/` and `data/eval_labels/`):
    - Embeds each utterance as `"<speaker>: <text>"` via `text-embedding-3-large`.
    - Caches per-transcript embeddings as `embeddings_cache/<stem>.npy` (float32, shape
      `(n_utterances, 3072)`). Cache files are gitignored and reproducible from the API.
    - `build_windowed_features` supports two modes, both producing feature dim =
      `(2×WINDOW+1) × 3 072 = 15 360`: default index-based (±`WINDOW` nearest by position,
      zero-padded at boundaries) and timestamp-based (up to `WINDOW` neighbors within N seconds on
      each side, zero-padded if fewer qualify). Mode is selected via `--timestamp-window`.
    - `get_utterance_seconds` returns a float32 array of per-utterance timestamps (seconds) for a
      transcript; used by timestamp mode.
    - `examples_to_Xy` returns `(X, y, source_files, matched_examples)` — the 4th value threads
      the original example dicts through for error analysis.
    - Trains `LogisticRegression(class_weight="balanced")` on the training labels, saves the
      classifier to `results/embed_model/classifier.joblib`, then evaluates on the eval labels.
      Writes classification report, confusion matrix, error analysis, and per-conversation F1 to
      `evaluation_embed/`.

13. **`predict_embed.py`** — inference + bootstrap-labeling tool for the OpenAI embedding pipeline:
    - Loads the saved classifier from `results/embed_model/classifier.joblib`.
    - `predict_file` embeds a transcript (cached), builds windowed features, and returns per-utterance
      predictions with confidence and per-class probabilities.
    - `review_predictions` is an interactive reviewer: prints each utterance predicted as the positive
      class with its ±2 neighbor context and lets you accept or override from the keyboard.
    - `save_label_json` writes confirmed labels to `data/labels/<transcript_stem>.json`.
    - `get_unlabeled_files` finds transcripts without a matching label file.
    - CLI: `--file <path>` for a single transcript, `--auto` to skip review.

14. **`bert_embed.py`** — BERT embedding pipeline + logistic regression classifier. Uses frozen
    `bert-base-uncased` as a feature extractor (mean-pooled last hidden state, 768-dim). Same
    structure as `embed.py` but with no API key required. Caches to
    `bert_embeddings_cache/<stem>.npy`. Feature dim = 3 840 (`WINDOW = 2`, 5 × 768).
    `build_windowed_features` and `get_utterance_seconds` mirror `embed.py`; feature dim =
    `(2×WINDOW+1) × 768 = 3 840` in both modes. `examples_to_Xy` returns 4 values (same as `embed.py`).
    Saves the classifier to `results/bert_embed_model/classifier.joblib`. Writes classification
    report, confusion matrix, error analysis, and per-conversation F1 to `evaluation_bert_embed/`.

15. **`predict_bert_embed.py`** — inference + bootstrap-labeling tool for the BERT embedding
    pipeline. Same interface as `predict_embed.py` but loads from
    `results/bert_embed_model/classifier.joblib` and embeds via frozen `bert-base-uncased` instead
    of the OpenAI API.

## Working in this codebase

- **Label scheme** — two annotation sets exist in this repo, each with its own label directories
  and label JSON files. `label_schema.py` is the single source of truth and must match whichever
  set you are working with:
  - **O/I scheme** (`{"O": 0, "I": 1}`) — Outside / Inside goal discussion
  - **U/R scheme** (`{"U": 0, "R": 1}`) — Unrelated / Related to goal discussion (currently active)
  All label names, IDs, and defaults are defined in `label_schema.py` and imported everywhere else
  — no other file hardcodes label strings. To switch schemes, edit only `label_schema.py`:
  1. Change `LABEL2ID` (e.g. `{"O": 0, "I": 1}`).
  2. Set `DEFAULT_LABEL` to the new negative class (e.g. `"O"`).
  3. Set `POSITIVE_LABEL` to the new positive class (e.g. `"I"`).
  4. `ID2LABEL`, `LABEL_NAMES`, and `LABEL_IDS` derive automatically.
  5. Update the B/E legacy collapse target if those old annotations should map to the new positive
     label — `load_labels_from_json` in `prepare_data.py` uses `POSITIVE_LABEL` for this.
  6. Retrain from scratch (`python train.py`) — existing checkpoints are incompatible.
- **Utterance index space** — indices are 0-based positions among utterances only, not among all
  events. `print_indices.py`, `make_windowed_examples`, `predict_file`, and `load_labels_from_json`
  all use this convention. Never index by position in the full `events` list.
- The example `text` format (`[SEP]`-joined context with `[TARGET]` and `[APP_CTX]`) is defined in
  `prepare_data.make_windowed_examples` and consumed as-is by `train.py`, `evaluate.py`, and
  `predict.py`. The embedding pipelines also store this text in example dicts for error analysis
  display — keep all consumers in sync if you change it.
- `evaluate.py` and `predict.py` expect a model checkpoint at `./results/best_model/`. `train.py`
  writes there automatically at the end of training.
- `train.py`'s data loading and training loop live inside `if __name__ == "__main__":` — keep it
  that way. `evaluate.py` and `predict.py` both import `MAX_LENGTH` from `train`; any code at
  `train` import time would fire as a side effect.
- `evaluate.py` and `predict.py` run their model-loading/inference logic inside
  `if __name__ == "__main__":`. Nothing imports either of them, but this guard keeps them safe.
- All data files (`data/`, `results*/`, `evaluation*/`, `embeddings_cache/`,
  `bert_embeddings_cache/`) are gitignored. Never commit transcripts, labels, embeddings, or model
  weights.
