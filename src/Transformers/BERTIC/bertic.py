#Importing stuff
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# Seed (to ensure reproducibility)
set_seed(42)

dataset = load_dataset("csv", data_files={
    "train": "/home/sbiskup/OPJ/TRAIN.csv",
    "test": "/home/sbiskup/OPJ/test-1.csv"
})

full_train = dataset["train"].train_test_split(test_size=0.05, seed=12345)
dataset_train = full_train["train"]
dataset_valid = full_train["test"]


def tokenize(batch):
    return tokenizer(batch["Sentence"], padding="max_length", truncation=True, max_length=128)

model_name = "classla/bcms-bertic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


dataset_train = dataset_train.map(tokenize, batched=True)
dataset_valid = dataset_valid.map(tokenize, batched=True)
dataset["test"] = dataset["test"].map(tokenize, batched=True)


dataset_train = dataset_train.rename_column("Label", "labels")
dataset_valid = dataset_valid.rename_column("Label", "labels")
dataset["test"] = dataset["test"].rename_column("Label", "labels")

dataset_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dataset_valid.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dataset["test"].set_format("torch", columns=["input_ids", "attention_mask", "labels"])


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


training_args = TrainingArguments(
    output_dir="./bertic-sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    metric_for_best_model="f1",
    greater_is_better=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

# Save the model
model.save_pretrained("bertic-sentiment")
tokenizer.save_pretrained("bertic-sentiment")
