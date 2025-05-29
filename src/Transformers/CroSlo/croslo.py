# Importing stuff
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
from transformers import EarlyStoppingCallback
import pandas as pd

set_seed(12345)

#Configuring and out_dir creation
num_epochs = 4
batch_size = 16
learning_rate = 2e-5
max_seq_length = 128
model_name = "EMBEDDIA/crosloengual-bert"
out_dir = "result_croslo-TRAIN_PROBA"
os.makedirs(out_dir, exist_ok=True)

dataset = load_dataset("csv", data_files={
    "train": "/home/sbiskup/OPJ/TRAIN.csv",
    "test_1": "/home/sbiskup/OPJ/test-1.csv",
})

test_2 = load_dataset("csv", data_files={"test_2": "/home/sbiskup/OPJ/test-2.csv"})["test_2"]
test_3 = load_dataset("csv", data_files={"test_3": "/home/sbiskup/OPJ/test-3.csv"})["test_3"]

# Spliting the train dataset (for validation, 5%)
full_train = dataset["train"].train_test_split(test_size=0.05, seed=12345)
dataset_train = full_train["train"]
dataset_valid = full_train["test"]

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
dataset["test_1"] = dataset["test_1"].map(tokenize_function, batched=True)
test_2 = test_2.map(tokenize_function, batched=True)
test_3 = test_3.map(tokenize_function, batched=True)

dataset_train = dataset_train.rename_column("Label", "labels")
dataset_valid = dataset_valid.rename_column("Label", "labels")
dataset["test_1"] = dataset["test_1"].rename_column("Label", "labels")
test_2 = test_2.rename_column("Label", "labels")
test_3 = test_3.rename_column("Label", "labels")

dataset_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset["test_1"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_2.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_3.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

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
    weight_decay=0.02,
    logging_dir=f"{out_dir}/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

print("\nTraining Evaluation:")
train_metrics = trainer.evaluate(dataset_train)
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nValidation Evaluation:")
val_metrics = trainer.evaluate(dataset_valid)
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set 1 Evaluation (Group 1):")
test_1_metrics = trainer.evaluate(dataset["test_1"])
for k, v in test_1_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set 2 Evaluation (Group 2):")
test_2_metrics = trainer.evaluate(test_2)
for k, v in test_2_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set 3 Evaluation (Group 3 - Us):")
test_3_metrics = trainer.evaluate(test_3)
for k, v in test_3_metrics.items():
    print(f"{k}: {v:.4f}")

trainer.save_model(f"{out_dir}/best_model")
tokenizer.save_pretrained(f"{out_dir}/best_model")


