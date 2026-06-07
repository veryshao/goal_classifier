import json
import glob
import sys
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from prepare_data import load_all_data, make_windowed_examples
from parse_transcript import parse_transcript
from label_schema import LABEL2ID, ID2LABEL

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = "./results/best_model"
MAX_LENGTH = 384
REVIEW_LABELS = {"B", "E"}

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ── Dataset ───────────────────────────────────────────────────────────────────
class GoalDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = tokenizer(ex["text"], max_length=MAX_LENGTH,
                        padding="max_length", truncation=True,
                        return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(ex["label_id"], dtype=torch.long)
        }

# ── Predict on a single file ──────────────────────────────────────────────────

def predict_file(transcript_path: str) -> list[dict]:
    """
    Run the model on one transcript file.
    Returns a list of dicts, one per utterance, with prediction and context.
    """
    events = parse_transcript(transcript_path)

    # Build examples with all labels set to O (placeholder — we're predicting)
    utterance_indices = [i for i, e in enumerate(events)
                         if e.event_type == "utterance"]
    placeholder_labels = {i: "O" for i in utterance_indices}

    examples = make_windowed_examples(events, placeholder_labels)

    results = []
    for ex in examples:
        enc = tokenizer(
            ex["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**enc).logits

        probs     = torch.softmax(logits, dim=-1).squeeze().tolist()
        pred_id   = int(torch.argmax(logits, dim=-1).item())
        pred_label = ID2LABEL[pred_id]

        results.append({
            "utterance_idx": ex["event_idx"],
            "timestamp":     ex["timestamp"],
            "pred":          pred_label,
            "confidence":    round(max(probs), 3),
            "probs":         {ID2LABEL[i]: round(p, 3) for i, p in enumerate(probs)},
            "text":          ex["text"],   # full windowed input
        })

    return results


def get_target_utterance(windowed_text: str) -> str:
    """Extract just the [TARGET] utterance from the windowed input string."""
    for part in windowed_text.split("[SEP]"):
        if "[TARGET]" in part:
            return part.replace("[TARGET]", "").strip()
    return windowed_text[:120]


# ── Interactive review of B and E predictions ─────────────────────────────────

def review_predictions(results: list[dict],
                        transcript_path: str) -> dict[str, str]:
    """
    Print each predicted B or E for human review.
    Returns a label dict ready to save as a label JSON.
    """
    flagged = [r for r in results if r["pred"] in REVIEW_LABELS]
    confirmed_labels = {}

    if not flagged:
        print("  No B or E predictions — all utterances predicted O.")
        return confirmed_labels

    print(f"\n  {len(flagged)} utterance(s) predicted as B or E. Review each:\n")

    for r in flagged:
        target = get_target_utterance(r["text"])

        # Print context: the full window around the target
        print("  " + "─" * 70)
        print(f"  [{r['timestamp']}]  Predicted: {r['pred']}  "
              f"(confidence: {r['confidence']})")
        print(f"  Probabilities: { {k: v for k, v in r['probs'].items()} }")
        print(f"\n  Context window:")
        for part in r["text"].split("[SEP]"):
            part = part.strip()
            if not part:
                continue
            marker = "  >>> " if "[TARGET]" in part else "      "
            print(f"{marker}{part.replace('[TARGET]', '').strip()}")

        # Prompt for confirmation
        print(f"\n  Accept prediction '{r['pred']}'?")
        print(f"  Enter label [B / I / E / O] or press Enter to accept '{r['pred']}': ",
              end="")
        user_input = input().strip().upper()

        if user_input == "":
            final_label = r["pred"]
        elif user_input in LABEL2ID:
            final_label = user_input
        else:
            print(f"  Unrecognized input — defaulting to '{r['pred']}'")
            final_label = r["pred"]

        if final_label != "O":
            confirmed_labels[str(r["utterance_idx"])] = final_label

        print()

    return confirmed_labels


# ── Save label JSON ───────────────────────────────────────────────────────────

def save_label_json(labels: dict, transcript_path: str, output_dir: str = "data/labels"):
    """Save confirmed labels as a JSON file matching the transcript filename."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Match naming convention: same stem as transcript, .json extension
    stem     = os.path.splitext(os.path.basename(transcript_path))[0]
    out_path = os.path.join(output_dir, f"{stem}.json")

    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  Saved: {out_path}  ({len(labels)} non-O label(s))")
    return out_path


# ── Batch mode: run on all unlabeled files ────────────────────────────────────

def get_unlabeled_files(transcript_dir: str, label_dir: str) -> list[str]:
    """Return transcript files that don't yet have a label JSON."""
    import os
    all_transcripts = sorted(glob.glob(f"{transcript_dir}/*.txt"))
    labeled_stems   = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(f"{label_dir}/*.json")
    }
    unlabeled = [
        t for t in all_transcripts
        if os.path.splitext(os.path.basename(t))[0] not in labeled_stems
    ]
    return unlabeled


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict goal conversation boundaries and review flagged utterances."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a single transcript file. If omitted, runs on all unlabeled files."
    )
    parser.add_argument(
        "--transcript_dir", type=str, default="data/transcripts",
        help="Directory containing transcript .txt files."
    )
    parser.add_argument(
        "--label_dir", type=str, default="data/labels",
        help="Directory containing existing label .json files."
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Skip interactive review and save raw predictions directly."
             " Use only if you trust the model enough to skip review."
    )
    args = parser.parse_args()

    # Single file or batch
    if args.file:
        files = [args.file]
    else:
        files = get_unlabeled_files(args.transcript_dir, args.label_dir)
        print(f"Found {len(files)} unlabeled transcript(s).\n")

    for i, fpath in enumerate(files):
        print(f"[{i+1}/{len(files)}] {fpath}")

        results = predict_file(fpath)
        print(f"  {len(results)} utterances parsed.")

        if args.auto:
            # Save all non-O predictions directly without review
            labels = {
                str(r["utterance_idx"]): r["pred"]
                for r in results if r["pred"] != "O"
            }
        else:
            labels = review_predictions(results, fpath)

        save_label_json(labels, fpath, args.label_dir)

    print("\nDone.")