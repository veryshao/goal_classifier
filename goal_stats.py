"""Descriptive statistics for goal conversations in hand-labeled sessions.

Both annotation sets in this repo are supported (see ANNOTATION_SCHEMES in
label_schema.py) and both are processed by default:
  OI — O/I (Outside/Inside a goal-setting discussion), positive label "I",
       from data/labels/ and data/eval_labels/
  UR — U/R (Unrelated/Related to goal discussion), positive label "R"
       marking goal utterances, from data/binary_labels/ and
       data/eval_binary_labels/
Results are written to <output-dir>/<scheme>/ (e.g. evaluation_goal_stats/OI/).

A goal conversation ("segment") is a maximal run of strictly consecutive
utterance indices carrying the scheme's positive label after filling unlisted
indices with its default label — a single negative utterance splits a run
into two segments.

Definitions:
- Utterance index space: position among utterances only, in parse order,
  matching print_indices.py and the label JSON keys. Never re-sorted.
- Segment duration = max(seconds) - min(seconds) over the segment's
  utterances (min/max rather than first/last: timestamps can be
  non-monotonic when transcripts interleave multiple student/tutor pairs).
  This is a lower bound — only utterance *start* times exist, so the last
  utterance's own speaking time is excluded and a 1-utterance segment has
  duration 0. n_utterances is reported alongside.
- Session duration = max - min seconds over utterances only (app events
  such as trailing inactivity would distort normalization).
- Normalized position = (t - session_start) / session_duration, clamped to
  [0, 1]; None if session duration is 0.
- Word count = len(text.split()) (whitespace tokens).
- tutor_student_ratio = tutor_words / student_words (None when student
  words = 0); tutor_share = tutor_words / (tutor_words + student_words)
  (bounded, None only when both are 0). Both are computed goal-only and
  whole-session.

Sessions whose label file is an empty {} are real observations with
0 segments and are included in frequency and whole-session statistics.
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parse_transcript import parse_transcript, TranscriptEvent
from prepare_data import load_labels_from_json
from label_schema import ANNOTATION_SCHEMES


def word_count(text: str) -> int:
    return len(text.split())


def find_goal_segments(labels: Dict[int, str],
                       n_utterances: int,
                       positive_label: str,
                       default_label: str) -> List[Tuple[int, int]]:
    """Maximal runs of consecutive indices carrying positive_label.

    Returns inclusive (start_idx, end_idx) tuples. Unlisted indices default
    to default_label, so any gap ends the run.
    """
    segments = []
    run_start = None
    for i in range(n_utterances):
        if labels.get(i, default_label) == positive_label:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            segments.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        segments.append((run_start, n_utterances - 1))
    return segments


def speaker_word_counts(utterances: List[TranscriptEvent],
                        start_idx: int = 0,
                        end_idx: int = None) -> Dict[str, int]:
    """{"Tutor": n, "Student": n} over utterances[start_idx : end_idx+1]."""
    if end_idx is None:
        end_idx = len(utterances) - 1
    counts = {"Tutor": 0, "Student": 0}
    for utt in utterances[start_idx:end_idx + 1]:
        if utt.speaker in counts:
            counts[utt.speaker] += word_count(utt.text)
    return counts


def ratio_and_share(tutor: int,
                    student: int) -> Tuple[Optional[float], Optional[float]]:
    ratio = tutor / student if student > 0 else None
    share = tutor / (tutor + student) if (tutor + student) > 0 else None
    return ratio, share


def _norm_pos(t: float, session_start: int,
              session_duration: int) -> Optional[float]:
    if session_duration <= 0:
        return None
    return min(1.0, max(0.0, (t - session_start) / session_duration))


def segment_stats(utterances: List[TranscriptEvent],
                  start_idx: int,
                  end_idx: int,
                  session_start: int,
                  session_duration: int) -> Dict[str, Any]:
    seg_utts = utterances[start_idx:end_idx + 1]
    seconds = [u.seconds for u in seg_utts]
    seg_min, seg_max = min(seconds), max(seconds)
    counts = speaker_word_counts(utterances, start_idx, end_idx)
    ratio, share = ratio_and_share(counts["Tutor"], counts["Student"])
    return {
        "start_idx":           start_idx,
        "end_idx":             end_idx,
        "n_utterances":        end_idx - start_idx + 1,
        "start_timestamp":     utterances[start_idx].timestamp,
        "end_timestamp":       utterances[end_idx].timestamp,
        "start_seconds":       seg_min,
        "end_seconds":         seg_max,
        "duration_seconds":    seg_max - seg_min,
        "norm_start":          _norm_pos(seg_min, session_start, session_duration),
        "norm_mid":            _norm_pos((seg_min + seg_max) / 2,
                                         session_start, session_duration),
        "tutor_words":         counts["Tutor"],
        "student_words":       counts["Student"],
        "tutor_student_ratio": ratio,
        "tutor_share":         share,
    }


def session_stats(transcript_fp: str, label_fp: str,
                  positive_label: str, default_label: str) -> Dict[str, Any]:
    stem = os.path.splitext(os.path.basename(transcript_fp))[0]
    events = parse_transcript(transcript_fp)
    utterances = [e for e in events if e.event_type == "utterance"]
    labels = load_labels_from_json(label_fp)

    n = len(utterances)
    out_of_range = [i for i in labels if i >= n]
    if out_of_range:
        print(f"WARNING: {stem}: label indices {out_of_range} exceed "
              f"{n} parsed utterances; ignoring them", file=sys.stderr)

    unexpected = {v for v in labels.values()
                  if v not in (positive_label, default_label)}
    if unexpected:
        print(f"WARNING: {stem}: label values {sorted(unexpected)} do not "
              f"belong to the {default_label}/{positive_label} scheme; "
              f"treating them as {default_label}", file=sys.stderr)

    seconds = [u.seconds for u in utterances]
    session_start = min(seconds)
    session_end = max(seconds)
    session_duration = session_end - session_start

    segments = [segment_stats(utterances, s, e, session_start, session_duration)
                for s, e in find_goal_segments(labels, n,
                                               positive_label, default_label)]

    sess_counts = speaker_word_counts(utterances)
    sess_ratio, sess_share = ratio_and_share(sess_counts["Tutor"],
                                             sess_counts["Student"])

    goal_tutor = sum(seg["tutor_words"] for seg in segments)
    goal_student = sum(seg["student_words"] for seg in segments)
    goal_ratio, goal_share = ratio_and_share(goal_tutor, goal_student)

    total_goal_s = sum(seg["duration_seconds"] for seg in segments)
    goal_utts = sum(seg["n_utterances"] for seg in segments)

    return {
        "stem":                        stem,
        "label_dir":                   os.path.dirname(label_fp),
        "n_utterances":                n,
        "session_start_seconds":       session_start,
        "session_end_seconds":         session_end,
        "session_duration_seconds":    session_duration,
        "n_segments":                  len(segments),
        "total_goal_duration_seconds": total_goal_s,
        "goal_utterances":             goal_utts,
        "goal_time_fraction":          (total_goal_s / session_duration
                                        if session_duration > 0 else None),
        "goal_utterance_fraction":     goal_utts / n if n > 0 else None,
        "session_tutor_words":         sess_counts["Tutor"],
        "session_student_words":       sess_counts["Student"],
        "session_tutor_student_ratio": sess_ratio,
        "session_tutor_share":         sess_share,
        "goal_tutor_words":            goal_tutor,
        "goal_student_words":          goal_student,
        "goal_tutor_student_ratio":    goal_ratio,
        "goal_tutor_share":            goal_share,
        "segments":                    segments,
    }


def summarize(values: List[Optional[float]]) -> Dict[str, float]:
    """{n, mean, median, std, min, max} over non-None values; {} if empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n":      len(vals),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


def aggregate_stats(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_segments = [seg for s in sessions for seg in s["segments"]]

    pooled = {}
    for scope in ("session", "goal"):
        tutor = sum(s[f"{scope}_tutor_words"] for s in sessions)
        student = sum(s[f"{scope}_student_words"] for s in sessions)
        ratio, share = ratio_and_share(tutor, student)
        pooled[scope] = {
            "tutor_words":         tutor,
            "student_words":       student,
            "tutor_student_ratio": ratio,
            "tutor_share":         share,
        }

    return {
        "n_sessions":              len(sessions),
        "n_sessions_with_goal":    sum(1 for s in sessions if s["n_segments"] > 0),
        "n_sessions_without_goal": sum(1 for s in sessions if s["n_segments"] == 0),
        "n_segments_total":        len(all_segments),
        "segments_per_session":    summarize([s["n_segments"] for s in sessions]),
        "segment_duration_seconds":
            summarize([seg["duration_seconds"] for seg in all_segments]),
        "segment_n_utterances":
            summarize([seg["n_utterances"] for seg in all_segments]),
        "segment_norm_start":      summarize([seg["norm_start"] for seg in all_segments]),
        "segment_norm_mid":        summarize([seg["norm_mid"] for seg in all_segments]),
        "goal_time_fraction":      summarize([s["goal_time_fraction"] for s in sessions]),
        "goal_utterance_fraction":
            summarize([s["goal_utterance_fraction"] for s in sessions]),
        "session_tutor_share":     summarize([s["session_tutor_share"] for s in sessions]),
        "goal_tutor_share":        summarize([s["goal_tutor_share"] for s in sessions]),
        "pooled_words_whole_session": pooled["session"],
        "pooled_words_goal_only":     pooled["goal"],
    }


SEGMENT_CSV_COLUMNS = [
    "stem", "segment_idx", "start_idx", "end_idx", "n_utterances",
    "start_timestamp", "end_timestamp", "start_seconds", "end_seconds",
    "duration_seconds", "norm_start", "norm_mid",
    "tutor_words", "student_words", "tutor_student_ratio", "tutor_share",
]

SESSION_CSV_COLUMNS = [
    "stem", "label_dir", "n_utterances", "session_duration_seconds",
    "n_segments", "total_goal_duration_seconds", "goal_utterances",
    "goal_time_fraction", "goal_utterance_fraction",
    "session_tutor_words", "session_student_words",
    "session_tutor_student_ratio", "session_tutor_share",
    "goal_tutor_words", "goal_student_words",
    "goal_tutor_student_ratio", "goal_tutor_share",
]


def _round(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


def format_summary(agg: Dict[str, Any]) -> str:
    lines = []
    lines.append("Goal conversation descriptive statistics")
    lines.append("=" * 55)
    lines.append(f"Sessions:            {agg['n_sessions']} "
                 f"({agg['n_sessions_with_goal']} with goal talk, "
                 f"{agg['n_sessions_without_goal']} without)")
    lines.append(f"Goal segments total: {agg['n_segments_total']}")
    lines.append("")

    def block(title, stats, fmt="{:.2f}"):
        lines.append(title)
        if not stats:
            lines.append("  (no data)")
            return
        lines.append("  " + "  ".join(
            f"{k}={fmt.format(stats[k]) if k != 'n' else stats[k]}"
            for k in ("n", "mean", "median", "std", "min", "max")))

    block("Segments per session:", agg["segments_per_session"])
    block("Segment duration (s):", agg["segment_duration_seconds"])
    block("Segment length (utterances):", agg["segment_n_utterances"])
    block("Goal time fraction of session:", agg["goal_time_fraction"], "{:.3f}")
    block("Goal utterance fraction of session:",
          agg["goal_utterance_fraction"], "{:.3f}")
    block("Segment position (normalized start):",
          agg["segment_norm_start"], "{:.3f}")
    block("Segment position (normalized midpoint):",
          agg["segment_norm_mid"], "{:.3f}")
    lines.append("")

    lines.append("Tutor vs student talk (word counts)")
    lines.append("-" * 55)
    for title, key in (("Whole session (pooled)", "pooled_words_whole_session"),
                       ("Goal segments only (pooled)", "pooled_words_goal_only")):
        p = agg[key]
        ratio = f"{p['tutor_student_ratio']:.2f}" \
            if p["tutor_student_ratio"] is not None else "n/a"
        share = f"{p['tutor_share']:.3f}" if p["tutor_share"] is not None else "n/a"
        lines.append(f"{title}: tutor={p['tutor_words']} "
                     f"student={p['student_words']} "
                     f"tutor:student={ratio} tutor_share={share}")
    block("Per-session tutor share (whole session):",
          agg["session_tutor_share"], "{:.3f}")
    block("Per-session tutor share (goal only):",
          agg["goal_tutor_share"], "{:.3f}")
    return "\n".join(lines)


def write_outputs(sessions: List[Dict[str, Any]],
                  agg: Dict[str, Any],
                  output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "per_segment.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_CSV_COLUMNS)
        writer.writeheader()
        for s in sessions:
            for i, seg in enumerate(s["segments"]):
                row = {"stem": s["stem"], "segment_idx": i}
                row.update({k: _round(seg[k]) for k in SEGMENT_CSV_COLUMNS
                            if k in seg})
                writer.writerow(row)

    with open(os.path.join(output_dir, "per_session.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SESSION_CSV_COLUMNS)
        writer.writeheader()
        for s in sessions:
            writer.writerow({k: _round(s[k]) for k in SESSION_CSV_COLUMNS})

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(format_summary(agg) + "\n")

    all_segments = [seg for s in sessions for seg in s["segments"]]

    norm_mids = [seg["norm_mid"] for seg in all_segments
                 if seg["norm_mid"] is not None]
    _, ax = plt.subplots(figsize=(5, 4))
    ax.hist(norm_mids, bins=np.linspace(0, 1, 21))
    ax.set_xlabel("Normalized position in session (segment midpoint)")
    ax.set_ylabel("Segments")
    ax.set_title("Where goal conversations occur")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segment_positions.png"), dpi=150)
    plt.close()

    durations = [seg["duration_seconds"] for seg in all_segments]
    _, ax = plt.subplots(figsize=(5, 4))
    ax.hist(durations, bins=20)
    ax.set_xlabel("Segment duration (s)")
    ax.set_ylabel("Segments")
    ax.set_title("Goal conversation durations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segment_durations.png"), dpi=150)
    plt.close()


def run_scheme(scheme_name: str,
               label_dirs: List[str],
               transcripts_dir: str,
               output_dir: str) -> None:
    scheme = ANNOTATION_SCHEMES[scheme_name]
    positive, default = scheme["positive_label"], scheme["default_label"]

    print(f"\n=== Scheme {scheme_name} "
          f"({default}/{positive}: {scheme['description']}) ===")

    label_by_stem: Dict[str, str] = {}
    for label_dir in label_dirs:
        for fp in sorted(glob.glob(os.path.join(label_dir, "*.json"))):
            stem = os.path.splitext(os.path.basename(fp))[0]
            if stem in label_by_stem:
                print(f"WARNING: {stem} labeled in multiple dirs; "
                      f"using {label_by_stem[stem]}", file=sys.stderr)
                continue
            label_by_stem[stem] = fp

    sessions = []
    for stem, label_fp in sorted(label_by_stem.items()):
        transcript_fp = os.path.join(transcripts_dir, stem + ".txt")
        if not os.path.exists(transcript_fp):
            print(f"WARNING: no transcript for label file {label_fp}; skipping",
                  file=sys.stderr)
            continue
        s = session_stats(transcript_fp, label_fp, positive, default)
        sessions.append(s)
        print(f"{stem}: {s['n_segments']} segment(s), "
              f"goal time {s['total_goal_duration_seconds']}s "
              f"of {s['session_duration_seconds']}s")

    if not sessions:
        sys.exit(f"No labeled sessions found for scheme {scheme_name} "
                 f"in {label_dirs}.")

    agg = aggregate_stats(sessions)
    scheme_output_dir = os.path.join(output_dir, scheme_name)
    write_outputs(sessions, agg, scheme_output_dir)

    print()
    print(format_summary(agg))
    print(f"\nWrote per_segment.csv, per_session.csv, summary.json, "
          f"summary.txt, and figures to {scheme_output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descriptive statistics of goal conversations "
                    "from hand-labeled sessions")
    parser.add_argument("--schemes", nargs="+",
                        choices=sorted(ANNOTATION_SCHEMES),
                        default=sorted(ANNOTATION_SCHEMES),
                        help="Annotation scheme(s) to process (default: all)")
    parser.add_argument("--label-dirs", nargs="+", default=None,
                        help="Override the scheme's label directories (only "
                             "valid with a single --schemes value); first dir "
                             "wins on duplicate stems")
    parser.add_argument("--transcripts", default="data/transcripts",
                        help="Directory of transcript .txt files")
    parser.add_argument("--output-dir", default="evaluation_goal_stats",
                        help="Base output directory; results go to "
                             "<output-dir>/<scheme>/")
    args = parser.parse_args()

    if args.label_dirs and len(args.schemes) != 1:
        parser.error("--label-dirs requires exactly one --schemes value")

    for scheme_name in args.schemes:
        label_dirs = args.label_dirs or \
            ANNOTATION_SCHEMES[scheme_name]["label_dirs"]
        run_scheme(scheme_name, label_dirs, args.transcripts, args.output_dir)
