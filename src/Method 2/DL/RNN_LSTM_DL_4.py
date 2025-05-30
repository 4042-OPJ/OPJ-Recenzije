import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.optim as optim
import classla
import optuna
from optuna.trial import Trial
from collections import Counter, defaultdict
import csv
import os
import random
import json


BATCH_SIZE = 128
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 3
DROPOUT_RATE = 0.3
LR = 5e-3
N_EPOCHS = 20
MIN_FREQ = 3
MAX_LENGTH = 256
FASTTEXT_PATH = "/home/fmadaric/opj/DLver3/cc.hr.300.vec"
SEED = 1234
MODEL_PATH = "RNN-T-t3.pt"
PARAMS_PATH = "RNNOptuna_params.json" #obsolete

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_N = 1
test_data_df = pd.read_csv("test-1.csv")
test_data = test_data_df.to_dict(orient='records')

test1_df = pd.read_csv("test-1.csv")
test2_df = pd.read_csv("test-2.csv")
test3_df = pd.read_csv("test-3.csv")

train1_df = pd.read_csv("train-1.csv")
train2_df = pd.read_csv("train-2.csv")
train3_df = pd.read_csv("train-3.csv")
#train_data_df = pd.concat([train1_df, train2_df, train3_df])
train_data_df = pd.read_csv("train-3.csv")

train_data_df = train_data_df.sample(frac=1).reset_index(drop=True)

all_test_df = pd.concat([test1_df, test2_df, test3_df])

nlp = classla.Pipeline(lang='hr', processors='tokenize', use_gpu=True)

def tokenize_hr(text):
    doc = nlp(text)
    return [token.text.lower() for sentence in doc.sentences for token in sentence.tokens]

def build_vocab(data, min_freq):
    freq = Counter(token for example in data for token in example["tokens"])
    vocab_tokens = {tok for tok, count in freq.items() if count >= min_freq}
    itos = ["<unk>", "<pad>"] + sorted(vocab_tokens)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos

class Vocab:
    def __init__(self, stoi, itos, unk_token="<unk>"):
        self.stoi = stoi
        self.itos = itos
        self.unk_index = stoi[unk_token]

    def lookup_indices(self, tokens):
        return [self.stoi.get(token, self.unk_index) for token in tokens]

    def __len__(self):
        return len(self.itos)

def load_fasttext_embeddings(filepath, stoi, embedding_dim):
    embeddings = np.random.uniform(-0.25, 0.25, (len(stoi), embedding_dim))
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)
        for line in f:
            parts = line.rstrip().split(' ')
            token = parts[0]
            if token in stoi:
                embeddings[stoi[token]] = np.array(parts[1:], dtype=np.float32)
    return torch.tensor(embeddings, dtype=torch.float32)

def preprocess(data, tokenizer, vocab=None):
    for example in data:
        tokens = tokenizer(example["Sentence"])[:MAX_LENGTH]
        example["tokens"] = tokens
        example["length"] = len(tokens)
        if vocab:
            example["ids"] = vocab.lookup_indices(tokens)

def pad_sequence(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    return torch.LongTensor([seq + [pad_idx] * (max_len - len(seq)) for seq in sequences])

def batch_iter(data, batch_size, pad_index, shuffle=True):
    if shuffle:
        random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        ids = [ex["ids"] for ex in batch]
        lengths = torch.LongTensor([len(seq) for seq in ids])
        ids_padded = pad_sequence(ids, pad_index)
        labels = torch.LongTensor([ex["Label"] for ex in batch])
        yield {"ids": ids_padded, "length": lengths, "label": labels}

max_length = MAX_LENGTH
min_freq = MIN_FREQ
batch_size = BATCH_SIZE
embedding_path = FASTTEXT_PATH

output_dim = 3  # 0=positive,1=neutral,2=negative
hidden_dim = HIDDEN_DIM
n_layers = N_LAYERS
bidirectional = True
dropout_rate = DROPOUT_RATE



train_data = train_data_df.to_dict(orient='records')
special_tokens = ["<unk>", "<pad>"]
preprocess(train_data, tokenize_hr)
preprocess(test_data, tokenize_hr)
stoi, itos = build_vocab(train_data, min_freq)
vocab = Vocab(stoi, itos)
unk_index = stoi["<unk>"]
pad_index = stoi["<pad>"]
preprocess(train_data, tokenize_hr, vocab)
preprocess(test_data, tokenize_hr, vocab)

vocab_size = len(stoi)

embedding_dim = EMBEDDING_DIM
pretrained_embeddings = load_fasttext_embeddings(embedding_path, stoi, EMBEDDING_DIM)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout_rate, pad_index, pretrained_embeddings=pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            bidirectional=bidirectional, dropout=dropout_rate if n_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = rnn_utils.pack_padded_sequence(embedded, length.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-2], hidden[-1]], dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

def get_predictions_and_labels(dataloader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["ids"].to(device)
            length = batch["length"]
            labels = batch["label"].to(device)
            outputs = model(ids, length)
            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

class SentimentDataset(Dataset):
    def __init__(self, ids_list, labels):
        self.ids_list = ids_list
        self.labels = labels

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        return {"ids": torch.tensor(self.ids_list[idx], dtype=torch.long),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
                "length": len(self.ids_list[idx])}

def collate_fn(batch):
    ids = [item["ids"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    padded_ids = rnn_utils.pad_sequence(ids, batch_first=True, padding_value=pad_index)
    return {"ids": padded_ids, "length": lengths, "label": labels}

def get_loader(ids, labels, batch_size=512):
    dataset = SentimentDataset(ids, labels)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

def objective(trial: Trial):
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    model = LSTM(
        vocab_size=len(stoi),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=hidden_dim,
        output_dim=3,
        n_layers=n_layers,
        bidirectional=True,
        dropout_rate=dropout_rate,
        pad_index=pad_index,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(N_EPOCHS):
        train_loss, _ = train_epoch(train_data, model, optimizer, criterion, device, pad_index)
        val_loss, _ = eval_epoch(test_data, model, criterion, device, pad_index)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    val_preds, val_labels = get_predictions_and_labels(get_loader([ex["ids"] for ex in test_data], [ex["Label"] for ex in test_data]), model, device)

    return f1_score(val_labels, val_preds, average="weighted")

def get_accuracy(prediction, label):
    return (prediction.argmax(dim=1) == label).sum().item() / len(label)

def train_epoch(data, model, optimizer, criterion, device, pad_index):
    model.train()
    losses, accuracies = [], []
    for batch in batch_iter(data, BATCH_SIZE, pad_index):
        ids, length, label = batch["ids"].to(device), batch["length"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        output = model(ids, length)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accuracies.append(get_accuracy(output, label))
    return np.mean(losses), np.mean(accuracies)

def eval_epoch(data, model, criterion, device, pad_index):
    model.eval()
    losses, accuracies = [], []
    with torch.no_grad():
        for batch in batch_iter(data, BATCH_SIZE, pad_index, shuffle=False):
            ids, length, label = batch["ids"].to(device), batch["length"].to(device), batch["label"].to(device)
            output = model(ids, length)
            loss = criterion(output, label)
            losses.append(loss.item())
            accuracies.append(get_accuracy(output, label))
    return np.mean(losses), np.mean(accuracies)

n_epochs = N_EPOCHS
best_valid_loss = float("inf")

# model = LSTM(
#     vocab_size,
#     embedding_dim,
#     hidden_dim,
#     output_dim,
#     n_layers,
#     bidirectional,
#     dropout_rate,
#     pad_index,
#     pretrained_embeddings=pretrained_embeddings
# )

if os.path.exists(MODEL_PATH):
    print(f"Model checkpoint found. Loading from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    saved_params = checkpoint["params"]

    best_model = LSTM(
        vocab_size=len(stoi),
        embedding_dim=saved_params["embedding_dim"],
        hidden_dim=saved_params["hidden_dim"],
        output_dim=saved_params["output_dim"],
        n_layers=saved_params["n_layers"],
        bidirectional=saved_params["bidirectional"],
        dropout_rate=saved_params["dropout_rate"],
        pad_index=pad_index,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    model = best_model

else:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial.params)

    best_params = study.best_trial.params
    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f)

    best_model = LSTM(
        vocab_size=len(stoi),
        embedding_dim=embedding_dim,
        hidden_dim=best_params['hidden_dim'],
        output_dim=output_dim,
        n_layers=best_params['n_layers'],
        bidirectional=True,
        dropout_rate=best_params['dropout_rate'],
        pad_index=pad_index,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)

    optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(N_EPOCHS):
        train_loss, _ = train_epoch(train_data, best_model, optimizer, criterion, device, pad_index)
        val_loss, _ = eval_epoch(test_data, best_model, criterion, device, pad_index)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = best_model.state_dict()

    best_model.load_state_dict(best_model_state)
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "params": {
            **best_params,
            "embedding_dim": embedding_dim,
            "output_dim": output_dim,
            "bidirectional": True
        }
    }, MODEL_PATH)
    model=best_model

def predict_sentiment(text, model, tokenizer, vocab, device):
    model.eval()
    tokens = tokenizer(text)[:MAX_LENGTH]
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    length = torch.LongTensor([len(ids)]).to(device)
    with torch.no_grad():
        output = model(tensor, length)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

def evaluate_and_plot(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted') * 100:.2f}%")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def preprocess_sentences(data, tokenizer, vocab, max_length=256):
    if isinstance(data, list):
        # lista dicta
        tokenized = [tokenizer(d["Sentence"])[:max_length] for d in data]
        ids = [vocab.lookup_indices(tokens) for tokens in tokenized]
        labels = [d["Label"] for d in data]
    else:
        # Za pd df
        tokenized = [tokenizer(text)[:max_length] for text in data["Sentence"]]
        ids = [vocab.lookup_indices(tokens) for tokens in tokenized]
        labels = data["Label"].tolist()
    return ids, labels


test3_ids, test3_labels = preprocess_sentences(test_data, tokenize_hr, vocab, max_length)
all_test_ids, all_test_labels = preprocess_sentences(all_test_df, tokenize_hr, vocab, max_length)

test3_loader = get_loader(test3_ids, test3_labels)
all_test_loader = get_loader(all_test_ids, all_test_labels)

test3_preds, test3_true = get_predictions_and_labels(test3_loader, model, device)
all_test_preds, all_test_true = get_predictions_and_labels(all_test_loader, model, device)

class_names = ["positive", "neutral", "negative"]

print("Test {} Results:".format(TEST_N))
evaluate_and_plot(test3_true, test3_preds, class_names)

#print("All Test Sets Combined Results:")
#evaluate_and_plot(all_test_true, all_test_preds, class_names)

