import os
import pandas as pd
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from src.config import HF_MODEL

def main():
    ds = load_dataset(
        "csv",
        data_files={"train": "data/train.csv", "validation": "data/dev.csv"}
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)

    def preprocess(batch):
        inputs = tokenizer(
            batch["diff"],
            truncation=True,
            max_length=1024
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["summary"],
                truncation=True,
                max_length=128
            )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)



    args = Seq2SeqTrainingArguments(
        output_dir="runs/finetune",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_dir="runs/logs",
        logging_steps=10,
        _n_gpu=1 if os.getenv("DEVICE", "cpu") == "mps" else 0
    )


    rouge = evaluate.load("rouge")



    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels]
        )

        return {"rougeL": result["rougeL"].mid.fmeasure}

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset= tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("runs/finetune/best_model")

if __name__ == "__main__":
    main()

