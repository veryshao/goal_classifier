import json
import os
from typing import List, Dict, Any
from parse_transcript import parse_transcript, TranscriptEvent
from label_schema import LABEL2ID

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
                           app_window_seconds: int = 30) -> List[Dict[str, Any]]:
    """
    For each labeled utterance, build a text window.
    Utterances are indexed by their position among utterances only,
    matching the output of print_indices.py.
    """
    utterance_events = [e for e in events if e.event_type == "utterance"]

    examples = []

    for utt_idx, utt in enumerate(utterance_events):
        if utt_idx not in labels:
            continue

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
                  label_files: List[str]) -> List[Dict[str, Any]]:
    """
    Match each transcript to its label file by filename stem (a transcript's
    .md.txt pairs with a same-stemmed .md.json label file) rather than by
    list position — transcript_files and label_files commonly differ in
    length and sort order, so zipping them positionally pairs the wrong
    sessions together. Transcripts without a label file are skipped.
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
        full_labels = {i: labels.get(i, "O") for i in range(len(utterance_events))}

        examples = make_windowed_examples(events, full_labels)

        for ex in examples:
            ex["source_file"] = transcript_fp
        all_examples.extend(examples)

    return all_examples


def load_labels_from_json(label_filepath: str) -> Dict[int, str]:
    """
    Load your hand-annotated labels. Expected format:
    {
      "15": "B",
      "16": "I",
      "17": "I",
      "34": "E"
    }
    Keys are event indices (as strings) from parse_transcript output.
    All unlisted utterances are assumed "O".
    """
    with open(label_filepath) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":
    import glob
    from label_schema import LABEL2ID

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
        if ex["label"] != "O":
            print(f"  label:     {ex['label']}")
            print(f"  timestamp: {ex['timestamp']}")
            print(f"  text:      {ex['text'][:200]}")
            break