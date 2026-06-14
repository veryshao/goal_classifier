import glob
import numpy as np
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, TrainingArguments, Trainer)
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch

from prepare_data import load_all_data
from label_schema import LABEL2ID, ID2LABEL

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
MAX_LENGTH   = 384   # longer than default because we include app context
BATCH_SIZE   = 8
EPOCHS       = 5

# ── Dataset ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

# ── Weighted trainer (handles O-label imbalance) ──────────────────────────
# Class weights must be computed per-fold from the *training* split only —
# computing them once from all_examples would leak the held-out conversation's
# label distribution into every fold's loss, undermining leave-one-out.
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    # ── Load data ─────────────────────────────────────────────────────────
    transcript_files = sorted(glob.glob("data/transcripts/*.txt"))
    label_files      = sorted(glob.glob("data/labels/*.json"))
    # load_all_data matches files by filename stem, not list position.

    all_examples = load_all_data(transcript_files, label_files)
    conversation_ids = list(set(ex["source_file"] for ex in all_examples))

    print(f"Loaded {len(all_examples)} utterances from {len(conversation_ids)} transcripts")

    label_counts = {}
    for ex in all_examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    print("Label distribution:", label_counts)

    # ── Leave-one-out training loop ───────────────────────────────────────
    fold_results = []

    for held_out in conversation_ids:
        train_ex = [ex for ex in all_examples if ex["source_file"] != held_out]
        val_ex   = [ex for ex in all_examples if ex["source_file"] == held_out]

        train_label_ids = [ex["label_id"] for ex in train_ex]
        weights = compute_class_weight("balanced",
                                        classes=np.array([0, 1]),
                                        y=np.array(train_label_ids))
        class_weights = torch.tensor(weights, dtype=torch.float)

        print(f"\n=== Held out: {held_out} | "
              f"Train: {len(train_ex)} | Val: {len(val_ex)} ===")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
        )

        args = TrainingArguments(
            output_dir=f"./results/{held_out.split('/')[-1]}",
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
            f1    = f1_score(labels, preds, average="macro", zero_division=0)
            print(classification_report(labels, preds,
                labels=[0, 1],
                target_names=["O", "I"], zero_division=0))
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
        fold_results.append({"held_out": held_out})

    print("\n=== LOO complete. Check per-fold classification reports above. ===")