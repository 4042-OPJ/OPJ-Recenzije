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

set_seed(42)

dataset = load_dataset("csv", data_files={
    "train": "/home/sbiskup/OPJ/TRAIN.csv",
    "test": "/home/sbiskup/OPJ/test-1.csv"
})

def tokenize(batch):
    return tokenizer(batch["Sentence"], padding="max_length", truncation=True, max_length=128)

model_name = "classla/bcms-bertic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "Label"])

encoded_dataset = encoded_dataset.rename_column("Label", "labels")


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
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("bertic-sentiment")
tokenizer.save_pretrained("bertic-sentiment")
