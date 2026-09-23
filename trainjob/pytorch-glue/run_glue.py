"""GLUE fine-tuning sample: a small pretrained model on SST-2 by default, HuggingFace Trainer.

torchrun sets RANK / WORLD_SIZE / LOCAL_RANK; Trainer switches to DDP automatically when WORLD_SIZE > 1.
Model and dataset are downloaded from the HuggingFace Hub (needs network or a mirror via HF_ENDPOINT).
"""
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": float((preds == labels).mean()), "f1": f1}


def main():
    model_name = os.environ.get("MODEL_NAME", "distilbert/distilbert-base-uncased")
    task = os.environ.get("TASK", "sst2")
    epochs = float(os.environ.get("EPOCHS", "3"))
    batch = int(os.environ.get("BATCH_SIZE", "32"))
    max_len = int(os.environ.get("MAX_SEQ_LENGTH", "128"))
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/out")
    if task not in TASK_KEYS:
        raise SystemExit(f"TASK must be one of {sorted(TASK_KEYS)}, got {task!r}")
    key1, key2 = TASK_KEYS[task]

    raw = load_dataset("nyu-mll/glue", task)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(rows):
        if key2 is None:
            return tokenizer(rows[key1], truncation=True, max_length=max_len)
        return tokenizer(rows[key1], rows[key2], truncation=True, max_length=max_len)

    tokenized = raw.map(tokenize, batched=True)
    num_labels = raw["train"].features["label"].num_classes
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=2e-5,
        eval_strategy="epoch",
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    if trainer.is_world_process_zero():
        print({k: round(v, 4) for k, v in metrics.items()}, flush=True)
        if os.environ.get("MLFLOW_TRACKING_URI"):
            try:
                import mlflow

                with mlflow.start_run(run_name=os.environ.get("KUBE_TRAINJOB_NAME", "glue-sample")):
                    mlflow.log_params({"model": model_name, "task": task, "epochs": epochs, "batch_size": batch})
                    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            except ImportError:
                print("mlflow not installed; skipping tracking", flush=True)
        print("done", flush=True)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
