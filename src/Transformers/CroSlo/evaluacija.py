
import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import csv

out_dir = "result_croslo/best_model"
test_csv_path = "/home/sbiskup/OPJ/test-3.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(out_dir)
model = AutoModelForSequenceClassification.from_pretrained(out_dir)
model.to(device)
model.eval()

dataset = load_dataset("csv", data_files={"test": test_csv_path})

# Renaming Labels collumn to labe (needed to work, model won't work otherwise)
dataset["test"] = dataset["test"].rename_column("Label", "labels")

def tokenize_function(examples):
    return tokenizer(
        examples["Sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,  
    )

dataset = dataset.map(tokenize_function, batched=True)
dataset["test"].set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Evaluating on test set...")
metrics = trainer.evaluate(dataset["test"])

print("\nEvaluation Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

metrics_path = os.path.join(out_dir, "eval_metrics.csv")
with open(metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for key, value in metrics.items():
        writer.writerow([key, value])

print(f"\nSaved evaluation metrics to {metrics_path}")

preds_output = trainer.predict(dataset["test"])
logits = preds_output.predictions
preds = logits.argmax(axis=-1)
labels = preds_output.label_ids

df_results = pd.DataFrame({
    "label": labels,
    "prediction": preds
})
predictions_path = os.path.join(out_dir, "test_predictions.csv")
df_results.to_csv(predictions_path, index=False)
print(f"Saved test predictions to {predictions_path}")