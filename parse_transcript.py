import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TranscriptEvent:
    timestamp: str
    seconds: int
    event_type: str
    speaker: Optional[str]
    text: str
    raw: str

def timestamp_to_seconds(ts: str) -> int:
    parts = ts.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])

# ── Patterns ──────────────────────────────────────────────────────────────────

UTTERANCE_RE = re.compile(
    r'\[(\d+:\d+)\]\s+(Tutor|Student)(?:\s+\([^)]+\))?:\s+([^\[]+)'
)

APP_EVENT_RE = re.compile(
    r'\[(\d+:\d+)\]\s+\[(app_switch|mouse click|mouse drag|keyboard type|'
    r'page_scroll|drawing|view_change|inactivity|system|mouse move|'
    r'keyboard delete|mouse hover) event[^\]]*\]\s*([^\[]*)'
)

HEADER_RE = re.compile(r'^\(school=')

def parse_transcript(filepath: str) -> List[TranscriptEvent]:
    """
    Parse a full transcript file into a flat list of TranscriptEvents.
    Reads the whole file as one string to handle transcripts where all
    content is on a single line.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # ── Strip inline annotator comments ──────────────────────────────────────────
    # Comments appear as plain text between transcript events, e.g.:
    # "Beginning of goal conversation" or "End of goal conversation"
    import re
    content = re.sub(r'\n?(Beginning of goal conversation|End of goal conversation)\n?', 
                    ' ', content)

    events = []

    # ── Collect all utterances ────────────────────────────────────────────────
    for m in UTTERANCE_RE.finditer(content):
        ts      = m.group(1)
        speaker = m.group(2)
        text    = m.group(3).strip()

        # Strip trailing event-like artifacts occasionally left at end of text
        text = re.sub(r'\s*\[\s*$', '', text).strip()
        if not text:
            continue

        events.append(TranscriptEvent(
            timestamp  = ts,
            seconds    = timestamp_to_seconds(ts),
            event_type = "utterance",
            speaker    = speaker,
            text       = text,
            raw        = m.group(0)
        ))

    # ── Collect all app/interaction events ───────────────────────────────────
    EVENT_TYPE_MAP = {
        "app_switch":     "app_switch",
        "mouse click":    "mouse_click",
        "mouse drag":     "mouse_drag",
        "keyboard type":  "keyboard_type",
        "keyboard delete":"keyboard_delete",
        "page_scroll":    "page_scroll",
        "drawing":        "drawing",
        "view_change":    "view_change",
        "inactivity":     "inactivity",
        "system":         "system",
        "mouse move":     "mouse_move",
        "mouse hover":    "mouse_hover",
    }

    for m in APP_EVENT_RE.finditer(content):
        ts       = m.group(1)
        raw_type = m.group(2)
        details  = m.group(3).strip()
        etype    = EVENT_TYPE_MAP.get(raw_type, "unknown")

        events.append(TranscriptEvent(
            timestamp  = ts,
            seconds    = timestamp_to_seconds(ts),
            event_type = etype,
            speaker    = None,
            text       = details,
            raw        = m.group(0)
        ))

    # ── Sort by timestamp ─────────────────────────────────────────────────────
    # Note: transcripts may have out-of-order timestamps due to multiple
    # student-tutor pairs being interleaved in one file. Sorting by seconds
    # gives the best approximation of chronological order, but is not perfect.
    events.sort(key=lambda e: e.seconds)

    return events


if __name__ == "__main__":
    # Quick test — run as: python parse_transcript.py data/conversations/session1.txt
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parse_transcript.py <transcript_file>")
        sys.exit(1)

    events = parse_transcript(sys.argv[1])
    utterances = [e for e in events if e.event_type == "utterance"]
    app_events = [e for e in events if e.event_type != "utterance"]

    print(f"\nParsed {len(events)} total events:")
    print(f"  {len(utterances)} utterances")
    print(f"  {len(app_events)} app/interaction events")
    print(f"\nFirst 5 utterances:")
    for e in utterances[:5]:
        print(f"  [{e.timestamp}] {e.speaker}: {e.text[:60]}")
    print(f"\nFirst 5 app events:")
    for e in app_events[:5]:
        print(f"  [{e.timestamp}] {e.event_type}: {e.text[:60]}")