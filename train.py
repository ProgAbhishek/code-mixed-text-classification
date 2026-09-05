"""
Fine-tune XLM-RoBERTa for Nepali-English code-mixed sentiment classification.

Usage:
    python scripts/train.py \
        --train data/train.tsv --val data/val.tsv \
        --model_name xlm-roberta-base --epochs 4 --batch_size 16

Expects TSV files with columns: id, text, sentiment
"""
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from preprocess import load_split, LABEL2ID, ID2LABEL


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_f1": f1, "precision": precision, "recall": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.tsv")
    ap.add_argument("--val", default="data/val.tsv")
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--output_dir", default="outputs/xlmr-sentiment")
    args = ap.parse_args()

    print(f"Loading data...")
    train_df = load_split(args.train)
    val_df = load_split(args.val)
    print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def to_hf_dataset(df):
        ds = Dataset.from_pandas(df[["text_clean", "label"]].rename(columns={"text_clean": "text"}))
        return ds

    train_ds = to_hf_dataset(train_df)
    val_ds = to_hf_dataset(val_df)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\nFinal validation metrics:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"\nModel saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
