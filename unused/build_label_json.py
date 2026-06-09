# build_label_json.py
# Fill in ANNOTATIONS below from your Google Doc, then run the script.
# One entry per labeled utterance — O's are filled in automatically.

import json

ANNOTATIONS = [
    # (event_index, label)   ← get event_index from print_indices.py output
    (6,  "B"),
    (7,  "I"),
    (8,  "I"),
    (9,  "I"),
    (10, "I"),
    (11, "E"),
]

output = {str(idx): label for idx, label in ANNOTATIONS}

with open("data/labels/session1_labels.json", "w") as f:
    json.dump(output, f, indent=2)

print("Written:", output)