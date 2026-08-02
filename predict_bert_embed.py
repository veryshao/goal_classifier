"""
Predict + bootstrap-label new transcripts using the BERT embedding pipeline.

Loads the saved LogisticRegression classifier from results/bert_embed_model/,
embeds each utterance via frozen bert-base-uncased (cached), builds windowed
features, and predicts. Mirrors predict.py's interactive review workflow.
"""
import argparse
import glob
import json
import os

import joblib

from parse_transcript import parse_transcript
from bert_embed import (get_or_cache_embeddings, get_utterance_seconds,
                        build_windowed_features, MODEL_SAVE_DIR)
from label_schema import LABEL2ID, DEFAULT_LABEL, POSITIVE_LABEL, LABEL_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(MODEL_SAVE_DIR, "classifier.joblib")
REVIEW_LABELS = {POSITIVE_LABEL}
CONTEXT_WINDOW = 2  # ±2 utterances shown during review

clf = joblib.load(MODEL_PATH)


# ── Predict on a single file ────────────────────────────────────────────────

def predict_file(transcript_path: str,
                 timestamp_window_seconds: int = None) -> list[dict]:
    """Run the BERT embedding classifier on one transcript. Returns a list of per-utterance dicts with prediction and context."""
    events     = parse_transcript(transcript_path)
    utterances = [e for e in events if e.event_type == "utterance"]

    if not utterances:
        return []

    embeddings = get_or_cache_embeddings(transcript_path)
    if timestamp_window_seconds is not None:
        ts       = get_utterance_seconds(transcript_path)
        features = build_windowed_features(embeddings, timestamps=ts,
                                           timestamp_window_seconds=timestamp_window_seconds)
    else:
        features = build_windowed_features(embeddings)

    probas = clf.predict_proba(features)
    preds  = clf.predict(features)
    class_labels = list(clf.classes_)

    results = []
    for i, (utt, pred, proba) in enumerate(zip(utterances, preds, probas)):
        results.append({
            "utterance_idx": i,
            "timestamp":     utt.timestamp,
            "speaker":       utt.speaker,
            "text":          utt.text,
            "pred":          pred,
            "confidence":    round(float(max(proba)), 3),
            "probs":         {cl: round(float(p), 3) for cl, p in zip(class_labels, proba)},
        })

    return results


# ── Interactive review ───────────────────────────────────────────────────────

def review_predictions(results: list[dict],
                       transcript_path: str) -> dict[str, str]:
    """Print each utterance predicted as positive for human review. Returns a label dict ready to save."""
    flagged = [r for r in results if r["pred"] in REVIEW_LABELS]
    confirmed_labels = {}

    if not flagged:
        print(f"  No {POSITIVE_LABEL} predictions — all utterances predicted {DEFAULT_LABEL}.")
        return confirmed_labels

    print(f"\n  {len(flagged)} utterance(s) predicted as {POSITIVE_LABEL}. Review each:\n")

    for r in flagged:
        idx = r["utterance_idx"]
        start = max(0, idx - CONTEXT_WINDOW)
        end   = min(len(results), idx + CONTEXT_WINDOW + 1)

        print("  " + "─" * 70)
        print(f"  [{r['timestamp']}]  Predicted: {r['pred']}  "
              f"(confidence: {r['confidence']})")
        print(f"  Probabilities: {r['probs']}")
        print(f"\n  Context window:")
        for j in range(start, end):
            ctx = results[j]
            marker = "  >>> " if j == idx else "      "
            print(f"{marker}[{ctx['timestamp']}] {ctx['speaker']}: {ctx['text']}")

        print(f"\n  Accept prediction '{r['pred']}'?")
        label_options = " / ".join(LABEL_NAMES)
        print(f"  Enter label [{label_options}] or press Enter to accept '{r['pred']}': ",
              end="")
        user_input = input().strip().upper()

        if user_input == "":
            final_label = r["pred"]
        elif user_input in LABEL2ID:
            final_label = user_input
        else:
            print(f"  Unrecognized input — defaulting to '{r['pred']}'")
            final_label = r["pred"]

        if final_label != DEFAULT_LABEL:
            confirmed_labels[str(idx)] = final_label

        print()

    return confirmed_labels


# ── Save label JSON ──────────────────────────────────────────────────────────

def save_label_json(labels: dict, transcript_path: str, output_dir: str = "data/labels"):
    """Save confirmed labels as a JSON file with the same stem as the transcript."""
    os.makedirs(output_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(transcript_path))[0]
    out_path = os.path.join(output_dir, f"{stem}.json")

    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"  Saved: {out_path}  ({len(labels)} non-{DEFAULT_LABEL} label(s))")
    return out_path


# ── Batch mode ───────────────────────────────────────────────────────────────

def get_unlabeled_files(transcript_dir: str, label_dir: str) -> list[str]:
    """Return transcript files that have no matching label JSON in label_dir."""
    all_transcripts = sorted(glob.glob(f"{transcript_dir}/*.txt"))
    labeled_stems   = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(f"{label_dir}/*.json")
    }
    return [
        t for t in all_transcripts
        if os.path.splitext(os.path.basename(t))[0] not in labeled_stems
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict goal conversation labels using the BERT embedding pipeline."
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
    )
    parser.add_argument(
        "--timestamp-window", type=int, default=None,
        help="Use timestamp-based feature window of N seconds. Must match the value "
             "used when training the classifier."
    )
    args = parser.parse_args()

    if args.file:
        files = [args.file]
    else:
        files = get_unlabeled_files(args.transcript_dir, args.label_dir)
        print(f"Found {len(files)} unlabeled transcript(s).\n")

    for i, fpath in enumerate(files):
        print(f"[{i+1}/{len(files)}] {fpath}")

        results = predict_file(fpath, timestamp_window_seconds=args.timestamp_window)
        print(f"  {len(results)} utterances parsed.")

        if args.auto:
            labels = {
                str(r["utterance_idx"]): r["pred"]
                for r in results if r["pred"] != DEFAULT_LABEL
            }
        else:
            labels = review_predictions(results, fpath)

        save_label_json(labels, fpath, args.label_dir)

    print("\nDone.")
