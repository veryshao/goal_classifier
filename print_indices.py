import sys
from parse_transcript import parse_transcript

if len(sys.argv) < 2:
    print("Usage: python print_indices.py data/transcripts/session1.txt")
    sys.exit(1)

filepath   = sys.argv[1]
events     = parse_transcript(filepath)
utterances = [e for e in events if e.event_type == "utterance"]

print(f"\nFile: {filepath}")
print(f"{'IDX':>5}  {'TIME':>8}  {'SPEAKER':>8}  TEXT")
print("-" * 80)

for i, utt in enumerate(utterances):
    print(f"{i:>5}  [{utt.timestamp}]  {utt.speaker:>8}  {utt.text}")
