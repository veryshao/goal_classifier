import re
import sys

UTTERANCE_RE = re.compile(r'\[(\d+(?::\d+)+)\]\s+(Tutor|Student)(?:\s+\([^)]+\))?:\s+([^\[]+)')

if len(sys.argv) < 2:
    print("Usage: python print_indices.py data/transcripts/session1.txt")
    sys.exit(1)

filepath = sys.argv[1]

# Read the whole file as one string, then find all utterances
with open(filepath, encoding="utf-8") as f:
    content = f.read()

matches = list(UTTERANCE_RE.finditer(content))

print(f"\nFile: {filepath}")
print(f"{'IDX':>5}  {'TIME':>6}  {'SPEAKER':>8}  TEXT")
print("-" * 80)

for i, m in enumerate(matches):
    timestamp = m.group(1)
    speaker   = m.group(2)
    text      = m.group(3).strip()
    print(f"{i:>5}  [{timestamp}]  {speaker:>8}  {text[:70]}")