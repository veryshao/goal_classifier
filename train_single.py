import glob
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, TrainingArguments

from prepare_data import load_all_data
from label_schema import LABEL2ID, ID2LABEL
from train import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS, GoalDataset, WeightedTrainer

# Trains a single bert-base-uncased classifier on a stratified train/val split
# of all labeled data (rather than train.py's leave-one-conversation-out loop),
# for a quick end-to-end run. Checkpoints land under ./results/single.
OUTPUT_DIR = "./results/single"

if __name__ == "__main__":
    transcript_files = sorted(glob.glob("data/transcripts/*.txt"))
    label_files      = sorted(glob.glob("data/labels/*.json"))

    all_examples = load_all_data(transcript_files, label_files)
    conversation_ids = set(ex["source_file"] for ex in all_examples)
    print(f"Loaded {len(all_examples)} utterances from {len(conversation_ids)} transcripts")

    label_counts = {}
    for ex in all_examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    print("Label distribution:", label_counts)

    label_ids = [ex["label_id"] for ex in all_examples]
    train_ex, val_ex = train_test_split(
        all_examples, test_size=0.15, random_state=42, stratify=label_ids
    )
    print(f"Train: {len(train_ex)} | Val: {len(val_ex)}")

    train_label_ids = [ex["label_id"] for ex in train_ex]
    weights = compute_class_weight("balanced",
                                    classes=np.array([0, 1, 2, 3]),
                                    y=np.array(train_label_ids))
    class_weights = torch.tensor(weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        print(classification_report(labels, preds,
            labels=[0, 1, 2, 3],
            target_names=["O", "B", "I", "E"], zero_division=0))
        return {"f1_macro": f1}

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=GoalDataset(train_ex),
        eval_dataset=GoalDataset(val_ex),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    print(f"\n=== Training complete. Checkpoints saved under {OUTPUT_DIR} ===")
