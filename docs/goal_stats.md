# Goal Conversation Statistics

`goal_stats.py` computes descriptive statistics of goal conversations from the
hand-annotated sessions: how often goal talk occurs, how long it lasts, who
does the talking (tutor vs. student word counts), and where in a session it
happens. It reads labels only — no model, API key, or GPU is required.

---

## Annotation schemes

Two parallel annotation sets exist in this repo, covering the same sessions:

| Scheme | Labels | Positive means | Label directories |
|--------|--------|----------------|-------------------|
| `OI` | `O` / `I` | utterance is **Inside** a goal-setting discussion (span-style: includes openings, asides, and closings within the span) | `data/labels/`, `data/eval_labels/` |
| `UR` | `U` / `R` | utterance is **Related** to goal discussion (per-utterance: only utterances that are themselves goal talk) | `data/binary_labels/`, `data/eval_binary_labels/` |

Both are described by `ANNOTATION_SCHEMES` in `label_schema.py`, independent
of which scheme is currently *active* for the training pipelines. Because U/R
marks individual goal utterances rather than whole spans, expect UR runs to
produce **more, shorter segments** than OI for the same sessions.

---

## Usage

```bash
# Both schemes (default) — writes evaluation_goal_stats/OI/ and .../UR/
python goal_stats.py

# One scheme only
python goal_stats.py --schemes OI

# Custom label directories (single scheme only)
python goal_stats.py --schemes OI --label-dirs data/eval_labels

# Other flags
python goal_stats.py --transcripts data/transcripts --output-dir evaluation_goal_stats
```

The script iterates the scheme's label files (the labeled set defines which
sessions are included), matches each to `data/transcripts/<same-stem>.txt`,
and warns about label files with no transcript, duplicate stems across label
directories, out-of-range label indices, and label values that don't belong
to the scheme.

---

## Definitions

- **Segment (goal conversation)** — a maximal run of strictly consecutive
  utterance indices carrying the scheme's positive label. Unlisted indices
  take the default label, so a single negative utterance splits a run into
  two segments.
- **Segment duration** — `max − min` of the segment's utterance timestamps
  (seconds). This is a **lower bound**: transcripts only record when each
  utterance *starts*, so the final utterance's own speaking time is not
  counted and a 1-utterance segment has duration 0. Check `n_utterances`
  alongside duration when interpreting short segments.
- **Session duration** — `max − min` timestamp over utterances only (app
  events are excluded; trailing inactivity events would distort it).
- **Normalized position** — `(t − session_start) / session_duration`,
  clamped to [0, 1]. 0 = session start, 1 = session end. Reported for each
  segment's start and midpoint.
- **Word count** — whitespace tokens (`text.split()`), summed per speaker.
- **`tutor_student_ratio`** — tutor words ÷ student words. Unbounded and
  undefined (empty cell / `null`) when the student says nothing.
- **`tutor_share`** — tutor words ÷ (tutor + student words). Bounded in
  [0, 1] and almost always defined; prefer it for averaging across sessions.
  0.5 means balanced talk; 1.0 means the tutor spoke exclusively.

Sessions whose label file is an empty `{}` are real observations: they appear
with `n_segments = 0` and count toward frequency statistics.

---

## Outputs (`evaluation_goal_stats/<scheme>/`)

| File | Contents |
|------|----------|
| `per_segment.csv` | One row per goal segment: indices, timestamps, duration, normalized positions, per-speaker word counts, ratio/share |
| `per_session.csv` | One row per labeled session: segment count, total goal time, goal time/utterance fractions, whole-session and goal-only word counts, ratio/share |
| `summary.json` | Corpus-level aggregates (machine-readable) |
| `summary.txt` | The same aggregates, human-readable (also printed to stdout) |
| `segment_positions.png` | Histogram of segment midpoints (normalized 0–1) |
| `segment_durations.png` | Histogram of segment durations (seconds) |

---

## Interpreting the summary

- **Sessions with/without goal talk** — frequency of goal conversations at
  the session level. A session "without" simply has an empty label file.
- **Segments per session** — the mean includes zero-segment sessions; the
  median being 0 while the mean is ~1–2 means goal talk is concentrated in
  roughly half the sessions.
- **Segment duration / length** — how long a typical goal conversation runs,
  in seconds and utterances (duration is a lower bound; see above).
- **Goal time / utterance fraction** — share of each session spent on goal
  talk.
- **Segment position** — where goal talk happens: values near 0 mean the
  start of the session. Compare the mean/median with `segment_positions.png`
  to see whether goal talk clusters early, late, or is bimodal.
- **Pooled vs. per-session tutor share** — pooled counts weight long sessions
  more; the per-session summary shows the distribution across sessions. If
  goal-only tutor share exceeds whole-session share, tutors dominate goal
  discussions more than the rest of the session (typical: goal-setting is
  tutor-led).
- **No-goal sessions** — the block reports tutor-vs-student word counts pooled
  over sessions whose label file has zero goal utterances, plus the
  distribution of per-session tutor share within that subset. Comparing this
  `tutor_share` against the whole-session average shows whether the balance
  of talk differs between sessions that included goal-setting and those that
  didn't. Printed as `(no sessions without goal talk)` if the entire corpus
  subset has goal utterances.
- **Non-goal utterances (all sessions)** — word counts pooled across every
  utterance that falls outside a goal segment, regardless of whether the
  session contains goal talk. Comparing this `tutor_share` against the
  goal-only share shows how much more (or less) tutor-dominated goal talk is
  relative to the rest of the corpus.
- **Non-goal utterances within goal-containing sessions** — same calculation
  restricted to sessions that have at least one goal segment. This isolates
  the "non-goal portion" of sessions where goals were actually discussed,
  enabling a within-session comparison: how does the tutor/student balance
  during goal talk compare with the balance in the same sessions when the
  conversation has moved on?
- **OI vs. UR** — comparing the two schemes' outputs shows how much of an
  O/I goal *span* is actual goal talk (UR) versus surrounding transition
  utterances included in the span.
