"""Builds windowed utterance examples from parsed transcripts and sparse label dicts.

Core functions:
  get_app_context_around — collects non-utterance events within ±30 seconds of a
    target and formats them as [SCREEN:], [CLICK:], [DRAG:] etc. tags.
  make_windowed_examples — builds one example per utterance: a [SEP]-joined context
    window with the target wrapped in [TARGET]...[/TARGET] and an [APP_CTX] suffix.
    Two windowing modes: utterance_window=N (±N neighbors by position) or
    timestamp_window_seconds=N (all utterances within N seconds).
  load_labels_from_json  — reads a sparse label JSON; maps legacy B/E values to
    POSITIVE_LABEL.
  load_all_data          — stem-matches transcript .txt files with label .json files,
    skips unmatched pairs, and returns a list of example dicts.
"""
import json
import os
from typing import List, Dict, Any
from parse_transcript import parse_transcript, TranscriptEvent
from label_schema import LABEL2ID, DEFAULT_LABEL, POSITIVE_LABEL

# Concise tag used to summarize each non-utterance event type in [APP_CTX].
# Covers every event_type produced by parse_transcript's EVENT_TYPE_MAP so
# none of them are silently dropped from the model's screen-context window.
EVENT_TAG = {
    "app_switch":      "SCREEN",
    "mouse_click":     "CLICK",
    "mouse_drag":      "DRAG",
    "keyboard_type":   "TYPE",
    "keyboard_delete": "DELETE",
    "page_scroll":     "SCROLL",
    "drawing":         "DRAW",
    "view_change":     "VIEW",
    "inactivity":      "IDLE",
    "system":          "SYSTEM",
    "mouse_move":      "MOVE",
    "mouse_hover":     "HOVER",
}

def get_app_context_around(events: List[TranscriptEvent],
                            utterance_idx: int,
                            window_seconds: int = 30) -> str:
    """
    For an utterance at events[utterance_idx], collect all non-utterance
    events within ±window_seconds and summarize them as a text string.
    This gives the model awareness of what's on screen.
    """
    center_time = events[utterance_idx].seconds
    app_lines = []

    for e in events:
        if e.event_type == "utterance":
            continue
        if abs(e.seconds - center_time) <= window_seconds:
            tag = EVENT_TAG.get(e.event_type)
            if tag:
                app_lines.append(f"[{tag}: {e.text}]")

    return " ".join(app_lines)

def make_windowed_examples(events: List[TranscriptEvent],
                           labels: Dict[int, str],
                           utterance_window: int = 2,
                           app_window_seconds: int = 30,
                           timestamp_window_seconds: int = None) -> List[Dict[str, Any]]:
    """
    For each labeled utterance, build a text window.
    Utterances are indexed by their position among utterances only,
    matching the output of print_indices.py.

    By default (timestamp_window_seconds=None) the context is the
    ±utterance_window nearest utterances by position. If
    timestamp_window_seconds is set, the context is instead all utterances
    within that many seconds of the target utterance's timestamp; in that
    case utterance_window is ignored for context selection.
    """
    utterance_events = [e for e in events if e.event_type == "utterance"]

    examples = []

    for utt_idx, utt in enumerate(utterance_events):
        if utt_idx not in labels:
            continue

        if timestamp_window_seconds is not None:
            before = [e for e in utterance_events[:utt_idx]
                      if abs(e.seconds - utt.seconds) <= timestamp_window_seconds]
            after  = [e for e in utterance_events[utt_idx + 1:]
                      if abs(e.seconds - utt.seconds) <= timestamp_window_seconds]
        else:
            before = utterance_events[max(0, utt_idx - utterance_window):utt_idx]
            after  = utterance_events[utt_idx + 1:utt_idx + 1 + utterance_window]

        context_parts = []
        for e in before:
            context_parts.append(f"{e.speaker}: {e.text}")
        context_parts.append(f"[TARGET] {utt.speaker}: {utt.text}")
        for e in after:
            context_parts.append(f"{e.speaker}: {e.text}")

        utterance_text = " [SEP] ".join(context_parts)

        # Find this utterance's position in the full event list for app context
        full_event_idx = next(
            i for i, e in enumerate(events)
            if e.event_type == "utterance" and e is utt
        )
        app_context = get_app_context_around(events, full_event_idx, app_window_seconds)

        full_text = utterance_text
        if app_context:
            full_text += " [APP_CTX] " + app_context

        examples.append({
            "text":        full_text,
            "label":       labels[utt_idx],
            "label_id":    LABEL2ID[labels[utt_idx]],
            "timestamp":   utt.timestamp,
            "event_idx":   utt_idx,
        })

    return examples


def load_all_data(transcript_files: List[str],
                  label_files: List[str],
                  timestamp_window_seconds: int = None) -> List[Dict[str, Any]]:
    """
    Match each transcript to its label file by filename stem (a transcript's
    .txt pairs with a same-stemmed .json label file) rather than by
    list position — transcript_files and label_files commonly differ in
    length and sort order, so zipping them positionally pairs the wrong
    sessions together. Transcripts without a label file are skipped.

    timestamp_window_seconds: if set, passed to make_windowed_examples to use
    time-based context windowing instead of the default utterance-count window.
    """
    label_by_stem = {
        os.path.splitext(os.path.basename(f))[0]: f for f in label_files
    }

    all_examples = []
    for transcript_fp in transcript_files:
        stem = os.path.splitext(os.path.basename(transcript_fp))[0]
        label_fp = label_by_stem.get(stem)
        if label_fp is None:
            continue  # no hand-labels for this session yet

        events = parse_transcript(transcript_fp)
        labels = load_labels_from_json(label_fp)

        # Index by utterance position only, matching print_indices.py
        utterance_events = [e for e in events if e.event_type == "utterance"]
        full_labels = {i: labels.get(i, DEFAULT_LABEL) for i in range(len(utterance_events))}

        examples = make_windowed_examples(events, full_labels,
                                          timestamp_window_seconds=timestamp_window_seconds)

        for ex in examples:
            ex["source_file"] = transcript_fp
        all_examples.extend(examples)

    return all_examples


def load_labels_from_json(label_filepath: str) -> Dict[int, str]:
    """
    Load hand-annotated labels.  Expected format: {"15": "I", "16": "I", ...}
    Keys are utterance indices (strings) from print_indices.py output.
    All unlisted utterances default to "O".
    Legacy B/E values from the old 4-class scheme are silently mapped to "I".
    """
    with open(label_filepath) as f:
        raw = json.load(f)
    _collapse = {"B": POSITIVE_LABEL, "E": POSITIVE_LABEL}
    return {int(k): _collapse.get(v, v) for k, v in raw.items()}


if __name__ == "__main__":
    import glob

    transcript_files = sorted(glob.glob("data/transcripts/*.txt"))
    label_files      = sorted(glob.glob("data/labels/*.json"))

    print(f"Found {len(transcript_files)} transcripts and {len(label_files)} label files")

    all_examples = load_all_data(transcript_files, label_files)
    matched_files = len(set(ex["source_file"] for ex in all_examples))
    print(f"{matched_files} transcripts matched a label file by filename "
          f"({len(transcript_files) - matched_files} have no labels yet and were skipped)")

    print(f"Total utterances: {len(all_examples)}")

    label_counts = {}
    for ex in all_examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    print("Label distribution:", label_counts)

    print("\nSample example:")
    for ex in all_examples:
        if ex["label"] != DEFAULT_LABEL:
            print(f"  label:     {ex['label']}")
            print(f"  timestamp: {ex['timestamp']}")
            print(f"  text:      {ex['text'][:200]}")
            break