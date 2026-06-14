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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384   # longer than default because we include app context
BATCH_SIZE = 8
EPOCHS     = 5

# ── Dataset ───────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class GoalDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = tokenizer(ex["text"], max_length=MAX_LENGTH,
                        padding="max_length", truncation=True,
                        return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(ex["label_id"], dtype=torch.long),
        }

# ── Weighted trainer (handles O/I class imbalance) ───────────────────────────
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

# ── Training only runs when executed directly ─────────────────────────────────
# evaluate.py and predict.py import MAX_LENGTH / GoalDataset from this module;
# keeping the training code under __main__ prevents a full training run from
# firing as a side effect of those imports.
if __name__ == "__main__":
    transcript_files  = sorted(glob.glob("data/transcripts/*.txt"))
    train_label_files = sorted(glob.glob("data/labels/*.json"))
    eval_label_files  = sorted(glob.glob("data/eval_labels/*.json"))

    train_examples = load_all_data(transcript_files, train_label_files)
    eval_examples  = load_all_data(transcript_files, eval_label_files)

    n_train_src = len({ex["source_file"] for ex in train_examples})
    n_eval_src  = len({ex["source_file"] for ex in eval_examples})
    print(f"Train: {len(train_examples)} utterances from {n_train_src} transcripts")
    print(f"Eval:  {len(eval_examples)} utterances from {n_eval_src} transcripts")

    label_counts = {}
    for ex in train_examples:
        label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
    print("Training label distribution:", label_counts)

    # Class weights from training data only
    train_label_ids = [ex["label_id"] for ex in train_examples]
    weights = compute_class_weight("balanced",
                                   classes=np.array([0, 1]),
                                   y=np.array(train_label_ids))
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Class weights: O={weights[0]:.3f}  I={weights[1]:.3f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir="./results/training",
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
              labels=[0, 1], target_names=["O", "I"], zero_division=0))
        return {"f1_macro": f1}

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=GoalDataset(train_examples),
        eval_dataset=GoalDataset(eval_examples),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # Save the best checkpoint (by eval f1_macro) to the canonical location
    trainer.save_model("./results/best_model")
    print("\n=== Training complete. Model saved to results/best_model/ ===")
