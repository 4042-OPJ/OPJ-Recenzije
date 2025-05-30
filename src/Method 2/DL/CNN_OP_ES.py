import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from gensim.models.fasttext import load_facebook_vectors
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import copy
from sklearn.utils.class_weight import compute_class_weight
import random
# nltk.download("punkt_tab")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




EPOCHS = 100 #Ova verzija ima early stopper pa ce se sama zaustavit kad se model ne poboljsava
PATIENCE = 7 #nakon kolko epocha ce stat kad ne vidi improvement
NUM_FILTERS = 275
FILTER_SIZES = [3, 4, 5]
LR = 1e-3
MAX_SEN_LEN = 150
DROPOUT = 0.48
BATCH_SIZE = 32 # manje vazno al ono... ak ne 32, 64 za brze i smooth, 16 za spoorije al bolja generalizacija

EMBEDDING_DIM = 300 # vec treniran vocab na 300


train_3_or_TRAIN = "TRAIN"

if train_3_or_TRAIN == "3":
    embedding_matrix_path = "embedding_matrix_3.npy"
    vocab_path = "vocab_cnn_nltk_3.json"
    model_path = "sentiment_cnn_model_train3_T4EStop.pt"
elif train_3_or_TRAIN == "TRAIN":
    embedding_matrix_path = "embedding_matrix.npy"
    vocab_path = "vocab_cnn_nltk_TRAIN.json" #ili _nltk_3
    model_path = "sentiment_cnn_model_TRAIN_t13EStop.pt" #ili _train3


#embedding_matrix_path = "embedding_matrix_3.npy"
#vocab_path = "vocab_cnn_nltk_3.json" #ili _nltk_3
#model_path = "sentiment_cnn_model_train3.pt" #ili _train3
test_data = pd.read_csv("test-1.csv")


train1_df = pd.read_csv("train-1.csv")
train2_df = pd.read_csv("train-2.csv")
train3_df = pd.read_csv("train-3.csv")

#train_data = pd.concat([train1_df, train2_df, train3_df])
train_data = pd.read_csv("train-3.csv")

train_sentences, train_labels = train_data['Sentence'].values, train_data['Label'].values
test_sentences, test_labels = test_data['Sentence'].values, test_data['Label'].values


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

def tokenize(text):
    return word_tokenize(text.lower())

word_to_idx = {}
idx = 2 
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1

def save_vocab(word_to_idx, path="vocab.json"):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(word_to_idx, f)

def load_vocab(path="vocab.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_vocab(sentences):
    global idx
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1

if os.path.exists(vocab_path):
    print("Loading existing vocabulary")
    word_to_idx = load_vocab(vocab_path)
    word_to_idx = {k: int(v) for k, v in word_to_idx.items()}
    idx = max(word_to_idx.values()) + 1
else:
    print("Building new vocabulary")
    build_vocab(train_sentences)
    save_vocab(word_to_idx, vocab_path)

embedding_dim = EMBEDDING_DIM

if os.path.exists(embedding_matrix_path):
    print("Loading embedding matrix from file...")
    embedding_matrix = np.load(embedding_matrix_path)
else:
    print("Loading FastText model and building embedding matrix...")
    fasttext_model = load_facebook_vectors("/home/fmadaric/opj/DLver3/cc.hr.300.bin")

    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
    for word, i in word_to_idx.items():
        if word in fasttext_model:
            embedding_matrix[i] = fasttext_model[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    np.save(embedding_matrix_path, embedding_matrix)
    print("Saved embedding matrix to file.")

def encode_sentence(sentence, max_len=MAX_SEN_LEN):
    tokens = tokenize(sentence)
    ids = [word_to_idx.get(w, word_to_idx['<UNK>']) for w in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [word_to_idx['<PAD>']] * (max_len - len(ids))
    return ids

class ReviewDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = [encode_sentence(s) for s in sentences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), self.labels[idx]

train_dataset = ReviewDataset(train_sentences, train_labels)
test_dataset = ReviewDataset(test_sentences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class SentimentCNN(nn.Module):
    def __init__(self, embedding_matrix, output_dim=3, filter_sizes=FILTER_SIZES, num_filters=NUM_FILTERS):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(i, dim=2)[0] for i in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR, loss_fn=None, patience=PATIENCE):
    model.to(device)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                preds = model(x_batch)
                pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
                all_preds.extend(pred_labels)
                all_labels.extend(y_batch.numpy())
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Validation F1 Score: {val_f1:.4f}")

        # Check early stopping condition
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(y_batch.numpy())
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")



cnn_model = SentimentCNN(embedding_matrix)
if os.path.exists(model_path):
    print("Loading saved model...")
    cnn_model.load_state_dict(torch.load(model_path))
    cnn_model.to(device)
else:
    print("Training")
    weighted_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
    train_model(cnn_model, train_loader, test_loader, loss_fn=weighted_loss)
    torch.save(cnn_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

evaluate_model(cnn_model, test_loader)
