import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    model_path = "bertic-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_dataset = load_dataset("csv", data_files={"test": "/home/sbiskup/OPJ/test-3.csv"})["test"]

    def tokenize(example):
        return tokenizer(example["Sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_test = test_dataset.map(tokenize)
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask"])
    
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_labels = []

    for batch in torch.utils.data.DataLoader(tokenized_test, batch_size=32):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)

    all_labels = test_dataset["Label"]
    metrics = compute_metrics(all_preds, all_labels)

    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
