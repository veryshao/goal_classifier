# build_label_json.py
# Fill in ANNOTATIONS below using indices from print_indices.py output,
# then run the script. Only list utterances labeled "I" — everything else
# defaults to "O" on load. Output filename is hardcoded; rename it manually.

import json

ANNOTATIONS = [
    # (utterance_index, label)   ← get indices from print_indices.py
    (6,  "I"),
    (7,  "I"),
    (8,  "I"),
    (9,  "I"),
    (10, "I"),
    (11, "I"),
]

output = {str(idx): label for idx, label in ANNOTATIONS}

with open("data/labels/session1_labels.json", "w") as f:
    json.dump(output, f, indent=2)

print("Written:", output)
