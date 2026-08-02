"""One-off helper for creating a label JSON file from print_indices.py output.

Fill in the ANNOTATIONS list with (utterance_index, label) pairs from
print_indices.py output, then run: python build_label_json.py

Output filename is hardcoded to session1_labels.json — rename manually after
running. For ongoing labeling, prefer predict.py's interactive review mode.
"""
# build_label_json.py
# Fill in ANNOTATIONS below using indices from print_indices.py output,
# then run the script. Only list utterances with the positive label —
# everything else defaults to the negative class on load. Output filename
# is hardcoded; rename it manually.

import json
from label_schema import POSITIVE_LABEL

ANNOTATIONS = [
    # (utterance_index, label)   ← get indices from print_indices.py
    (6,  POSITIVE_LABEL),
    (7,  POSITIVE_LABEL),
    (8,  POSITIVE_LABEL),
    (9,  POSITIVE_LABEL),
    (10, POSITIVE_LABEL),
    (11, POSITIVE_LABEL),
]

output = {str(idx): label for idx, label in ANNOTATIONS}

with open("data/labels/session1_labels.json", "w") as f:
    json.dump(output, f, indent=2)

print("Written:", output)
