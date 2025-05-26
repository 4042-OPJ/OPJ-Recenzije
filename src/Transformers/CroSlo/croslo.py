import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

set_seed(12345)

num_epochs = 4
batch_size = 16
learning_rate = 2e-5
max_seq_length = 128
model_name = "EMBEDDIA/crosloengual-bert"
out_dir = "result_croslo"
os.makedirs(out_dir, exist_ok=True)

dataset = load_dataset("csv", data_files={
    "train": "/home/sbiskup/OPJ/TRAIN.csv",
    "test": "/home/sbiskup/OPJ/test-1.csv"
})

# Spliting the train dataset (for validation, 5%)
full_train = dataset["train"].train_test_split(test_size=0.05, seed=12345)
dataset_train = full_train["train"]
dataset_valid = full_train["test"]

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(example):
    return tokenizer(
        example["Sentence"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_seq_length
    )
dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_valid = dataset_valid.map(tokenize_function, batched=True)
dataset["test"] = dataset["test"].map(tokenize_function, batched=True)

dataset_train = dataset_train.rename_column("Label", "labels")
dataset_valid = dataset_valid.rename_column("Label", "labels")
dataset["test"] = dataset["test"].rename_column("Label", "labels")

dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

training_args = TrainingArguments(
    output_dir=f"{out_dir}/logs",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir=f"{out_dir}/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model(f"{out_dir}/best_model")
tokenizer.save_pretrained(f"{out_dir}/best_model")

preds_output = trainer.predict(dataset["test"])
preds = preds_output.predictions.argmax(axis=-1)
labels = dataset["test"]["labels"]

print(f"Test accuracy: {accuracy_score(labels, preds):.4f}")

import pandas as pd
df_results = pd.DataFrame({"label": labels, "prediction": preds})
df_results.to_csv(f"{out_dir}/test_predictions.csv", index=False)
