import glob
import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from collections import defaultdict

from prepare_data import load_all_data
from label_schema import LABEL2ID, ID2LABEL
from train import MAX_LENGTH

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR        = "./results/best_model"   # path to your saved model
# MAX_LENGTH is imported from train so tokenization always matches training.
OUTPUT_DIR       = "./evaluation"
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ── Load all data (same as train.py) ─────────────────────────────────────────
transcript_files = sorted(glob.glob("data/transcripts/*.txt"))
label_files      = sorted(glob.glob("data/labels/*.json"))
all_examples     = load_all_data(transcript_files, label_files)

# ── Run inference ─────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

all_preds, all_labels, all_examples_with_preds = [], [], []

for ex in all_examples:
    enc = tokenizer(ex["text"], max_length=MAX_LENGTH,
                    padding="max_length", truncation=True,
                    return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    pred_id   = torch.argmax(logits, dim=-1).item()
    pred_label = ID2LABEL[pred_id]

    all_preds.append(pred_id)
    all_labels.append(ex["label_id"])
    all_examples_with_preds.append({**ex, "pred": pred_label, "pred_id": pred_id})

# ── 1. Overall classification report ─────────────────────────────────────────
print("\n=== Classification Report ===")
report = classification_report(
    all_labels, all_preds,
    target_names=["O", "B", "I", "E"],
    zero_division=0
)
print(report)
with open(f"{OUTPUT_DIR}/classification_report.txt", "w") as f:
    f.write(report)

# ── 2. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
fig, ax = plt.subplots(figsize=(6,5))
disp = ConfusionMatrixDisplay(cm, display_labels=["O","B","I","E"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"Confusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

# ── 3. Errors grouped by type ─────────────────────────────────────────────────
# The most useful view: what did the model predict when it was wrong,
# and what does the surrounding text look like?

error_groups = defaultdict(list)   # (true_label, pred_label) → [examples]

for ex in all_examples_with_preds:
    if ex["label"] != ex["pred"]:
        key = (ex["label"], ex["pred"])
        error_groups[key].append(ex)

print("\n=== Error Analysis ===")
error_summary = {}

for (true_lbl, pred_lbl), exs in sorted(error_groups.items()):
    key_str = f"TRUE={true_lbl} → PRED={pred_lbl}"
    print(f"\n{key_str}  ({len(exs)} errors)")
    error_summary[key_str] = []

    for ex in exs:
        # Print the [TARGET] utterance only for readability
        target_line = next(
            (part for part in ex["text"].split("[SEP]") if "[TARGET]" in part),
            ex["text"][:120]
        )
        print(f"  [{ex['timestamp']}] {target_line.strip()[:100]}")
        print(f"    source: {ex['source_file']}")

        error_summary[key_str].append({
            "timestamp":   ex["timestamp"],
            "source_file": ex["source_file"],
            "target_text": target_line.strip(),
            "full_input":  ex["text"]
        })

with open(f"{OUTPUT_DIR}/error_analysis.json", "w") as f:
    json.dump(error_summary, f, indent=2)
print(f"\nFull error details saved to {OUTPUT_DIR}/error_analysis.json")

# ── 4. Per-conversation performance ───────────────────────────────────────────
# With LOO you care a lot about which conversations are harder than others

print("\n=== Per-Conversation F1 (macro) ===")
per_conv = defaultdict(lambda: {"labels": [], "preds": []})

for ex in all_examples_with_preds:
    per_conv[ex["source_file"]]["labels"].append(ex["label_id"])
    per_conv[ex["source_file"]]["preds"].append(ex["pred_id"])

conv_scores = {}
for conv, data in sorted(per_conv.items()):
    f1 = f1_score(data["labels"], data["preds"],
                  average="macro", zero_division=0)
    conv_scores[conv] = round(f1, 3)
    print(f"  {conv.split('/')[-1]:40s}  macro-F1 = {f1:.3f}")

with open(f"{OUTPUT_DIR}/per_conversation_f1.json", "w") as f:
    json.dump(conv_scores, f, indent=2)

# ── 5. Boundary detection specifically ───────────────────────────────────────
# B and E are the labels you actually care about most — report them separately

print("\n=== Boundary Label Deep Dive (B and E only) ===")
for target_label in ["B", "E"]:
    target_id   = LABEL2ID[target_label]
    true_pos    = sum(1 for t, p in zip(all_labels, all_preds) if t == target_id and p == target_id)
    false_neg   = sum(1 for t, p in zip(all_labels, all_preds) if t == target_id and p != target_id)
    false_pos   = sum(1 for t, p in zip(all_labels, all_preds) if t != target_id and p == target_id)
    precision   = true_pos / (true_pos + false_pos)  if (true_pos + false_pos)  else 0
    recall      = true_pos / (true_pos + false_neg)  if (true_pos + false_neg)  else 0
    f1          = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    print(f"  {target_label}  precision={precision:.2f}  recall={recall:.2f}  F1={f1:.2f}  "
          f"(TP={true_pos}, FN={false_neg}, FP={false_pos})")

print(f"\nAll evaluation outputs written to ./{OUTPUT_DIR}/")