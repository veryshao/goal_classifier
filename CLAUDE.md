# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project that classifies utterances in tutoring-session transcripts according to whether they
fall **Outside / at the Begin / Inside / at the End (O/B/I/E)** of a "goal-setting discussion" span. It's
a per-utterance sequence-tagging task built on top of a `bert-base-uncased` sequence classifier
(`transformers` + `torch`).

There is no build system, package manifest, or test suite — this is a small data-science script
collection run directly with `python <script>.py`. Dependencies (inferred from imports) include
`torch`, `transformers`, `scikit-learn`, `matplotlib`, `numpy`.

## Commands

- Inspect how a transcript parses: `python parse_transcript.py data/transcripts/<file>.md.txt`
- List utterances with their indices (used to figure out which index to put in a label file when
  hand-labeling): `python print_indices.py data/transcripts/<file>.md.txt`
- Inspect data loading / windowed examples end-to-end: `python prepare_data.py` — globs all
  transcript+label pairs, prints the label distribution, and shows a sample non-`O` example.
- Train (leave-one-conversation-out): `python train.py` — fine-tunes `bert-base-uncased` once per
  held-out transcript, writing per-fold checkpoints to `./results/<conversation_name>/`.
- Run evaluation against a trained model: `python evaluate.py` — expects a fine-tuned model at
  `./results/best_model` and writes a classification report, confusion matrix, error analysis, and
  per-conversation F1 to `./evaluation/`.
- Predict + bootstrap-label new sessions: `python predict.py [--file <path>] [--auto]` — runs the
  saved `./results/best_model` over transcripts (by default, every transcript in
  `data/conversations/` lacking a label file in `data/labels/`), interactively reviews flagged `B`/`E`
  predictions (unless `--auto` is passed), and writes confirmed labels back to `data/labels/`.

There is no lint/test command configured. To exercise an individual piece, run the relevant module's
`__main__` block directly (e.g. `parse_transcript.py` and `print_indices.py` both take a transcript path
as `argv[1]`).

## Pipeline / architecture

Data flows through these stages:

1. **`data/transcripts/*.md.txt`** — raw session transcripts. Each file starts with a metadata header
   line `(school=..., teacher=..., session_date=..., session_time=..., tutor=...)` followed by inline
   timestamped content all on one logical stream:
   - utterances: `[mm:ss] Tutor|Student (Name): text`
   - interaction/app events: `[mm:ss] [app_switch|mouse click|keyboard type|... event: details]`
   Filenames encode `School##_Teacher##_<date>_<time>_Stu_<id>` and a transcript's `.md.txt` pairs with
   a same-named `.md.json` label file.

2. **`parse_transcript.py`** — regex-based parser (`UTTERANCE_RE`, `APP_EVENT_RE`) that turns a
   transcript file into a flat, timestamp-sorted list of `TranscriptEvent` dataclass instances
   (`event_type` is one of `utterance` or an interaction type like `app_switch`/`mouse_click`/etc.).
   Note: transcripts can interleave multiple student/tutor pairs, so sorting by `seconds` is only an
   approximation of true chronological order.

3. **`print_indices.py`** — prints `[index] [timestamp] [speaker] text` for every utterance matched by
   `UTTERANCE_RE` in a transcript. The printed `index` is the value used as the key in label JSON files
   (see below) — this script is the human labeling aid.

4. **`data/labels/*.md.json`** — sparse span annotations: a JSON object mapping an utterance **index**
   (string, matching the order from `print_indices.py`/`UTTERANCE_RE`) to `"B"` (begin of a
   goal-discussion span) or `"E"` (end of one). An empty object `{}` means the session has no
   goal-discussion span; some files are entirely empty/not-yet-labeled. The full O/B/I/E sequence is
   expected to be derived by expanding these sparse begin/end markers across the utterance sequence
   (everything between a B and its matching E becomes `I`, everything else `O`).

5. **`build_label_json.py`** — a one-off templated helper: paste `(event_index, label)` pairs (read off
   from `print_indices.py`) into the `ANNOTATIONS` list and run it; it fills in `"O"` for everything
   else and writes the result to `data/labels/session1_labels.json`. Note the output filename is
   hardcoded — it's meant to be hand-edited per session, not run as a general batch tool.

6. **`label_schema.py`** — defines `LABEL2ID`/`ID2LABEL` for the four-class `O/B/I/E` scheme
   (`O=0, B=1, I=2, E=3`), human-annotation guidance for what counts as `B`/`E`/`I`/`O`, and two
   heuristic signal lists used to spot goal-discussion cues: `GOAL_APP_SIGNALS` (screen/app-event
   strings like `"PLUS Students"`, `"Update Goals button"`, slider drags) and `GOAL_VERBAL_SIGNALS`
   (keywords like `"goal"`, `"minutes"`, `"skill"`, `"share my screen"`).

7. **`prepare_data.py`** — turns parsed events + sparse labels into model-ready examples:
   - `get_app_context_around` collects non-utterance events within ±30s of an utterance and renders
     them as tags like `[SCREEN: ...]` / `[CLICK: ...]` / `[DRAG: ...]` / `[TYPE: ...]`.
   - `make_windowed_examples` builds, per labeled utterance, a `[SEP]`-joined window of `±2`
     surrounding utterances with the target wrapped in `[TARGET] ...`, appending an `[APP_CTX] ...`
     suffix with the screen-context string — this is the final `text` fed to the model.
   - `load_labels_from_json` reads the sparse `{"index": "B"|"E"|"I"}` label dict.
   - `load_all_data(transcript_files, label_files)` ties it together: parses each transcript, fills
     unlabeled utterances with `"O"`, builds windowed examples, and tags each with `source_file`
     (used for leave-one-out splitting). Returns dicts with `text`, `label`, `label_id`,
     `source_file`, `timestamp`, and `event_idx` keys.

8. **`train.py`** — fine-tunes `bert-base-uncased` (`AutoModelForSequenceClassification`, 4 labels)
   for the O/B/I/E task:
   - `GoalDataset` wraps examples for the HF `Trainer`, tokenizing to `MAX_LENGTH = 384`.
   - Computes `class_weights` via `compute_class_weight("balanced", ...)` to counter `O`-label
     imbalance, and uses a custom `WeightedTrainer` with weighted cross-entropy loss.
   - Runs **leave-one-conversation-out (LOO)** cross-validation: for each transcript held out in turn,
     trains on the rest and evaluates on it, tracking `f1_macro` and saving per-fold checkpoints to
     `./results/<conversation_name>/`. (`evaluate.py` separately expects a single chosen-best
     checkpoint copied/symlinked to `./results/best_model`.)

9. **`evaluate.py`** — loads all transcripts+labels via `load_all_data`, runs the saved model from
   `./results/best_model` over every example, and writes to `./evaluation/`:
   - `classification_report.txt` (per-class precision/recall/F1 over O/B/I/E)
   - `confusion_matrix.png`
   - `error_analysis.json` (misclassified examples grouped by `(true, pred)` label pair)
   - `per_conversation_f1.json` (macro-F1 per source transcript — useful for leave-one-out style analysis)
   - console deep-dive on `B`/`E` boundary precision/recall/F1 specifically, since correctly finding the
     *boundaries* of a goal-discussion span is the metric that matters most for this task.

10. **`predict.py`** — inference + active/bootstrap-labeling tool built on the saved
    `./results/best_model`:
    - `predict_file` runs the model over every utterance in a transcript (using
      `make_windowed_examples` with placeholder `"O"` labels) and returns predictions with confidence
      and full per-class probabilities.
    - `review_predictions` is an interactive human-in-the-loop reviewer: it prints each utterance
      predicted `B` or `E` with its context window and probabilities and lets you accept the model's
      label or override it from the keyboard (`B`/`I`/`E`/`O`).
    - `save_label_json` writes confirmed labels to `data/labels/<transcript_stem>.json`, matching the
      transcript's naming convention.
    - `get_unlabeled_files` finds transcripts in `--transcript_dir` (default `data/conversations`,
      though transcripts currently live in `data/transcripts`) without a corresponding label file.
    - CLI: `--file <path>` for a single transcript (default: batch over all unlabeled files),
      `--auto` to skip interactive review and save raw `B`/`E`/`I` predictions directly. This is the
      bootstrap loop for growing the labeled set: train → predict on unlabeled sessions →
      human-confirm flagged `B`/`E` candidates → retrain.

## Working in this codebase

- The example `text` format (`[SEP]`-joined context with a `[TARGET]` marker, plus an `[APP_CTX]`
  suffix) is defined in `prepare_data.make_windowed_examples` and consumed as-is by `train.py`,
  `evaluate.py`, and `predict.py` — keep them in sync if you change it (e.g. `evaluate.py` and
  `predict.py` both split on `"[SEP]"` and look for `"[TARGET]"` to recover the target utterance for
  display/error-analysis).
- `LABEL2ID`/`ID2LABEL` and the `["O","B","I","E"]` ordering must stay consistent across
  `label_schema.py`, `prepare_data.py`, `train.py`, `evaluate.py`, and `predict.py` — `evaluate.py`
  hardcodes `target_names=["O","B","I","E"]` and `labels=[0,1,2,3]` for the confusion matrix.
- Not every transcript has a corresponding label file (and some label files are empty/blank rather than
  `{}`) — `load_all_data` handles this by defaulting unlabeled utterances to `"O"`.
- `predict.py`'s default `--transcript_dir` (`data/conversations`) doesn't match where transcripts
  actually live (`data/transcripts`) — pass `--transcript_dir data/transcripts` explicitly, or fix the
  default, when running it in batch mode.
- `evaluate.py` and `predict.py` expect a single consolidated checkpoint at `./results/best_model`,
  but `train.py`'s LOO loop writes one checkpoint per held-out conversation under
  `./results/<conversation_name>/` — you need to pick/copy the best fold's checkpoint into
  `./results/best_model` yourself before running evaluation or prediction.
