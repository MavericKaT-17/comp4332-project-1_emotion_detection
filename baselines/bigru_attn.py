import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DEVICE      =     "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
VOCAB_SIZE  = 30_000
EMB_DIM     = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
MAX_LEN     = 50
EPOCHS      = 40
BATCH_SIZE  = 128
LR          = 1e-3
DROPOUT     = 0.3
SEED        = 42
PATIENCE    = 8
CLIP_NORM   = 1.0
LABEL_SMOOTHING = 0.0
RUN_SWEEP   = True
SWEEP_CONFIGS = [
    {
        "name": "cfg_a",
        "emb_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "max_len": 64,
        "epochs": 40,
        "batch_size": 128,
        "lr": 1e-3,
        "dropout": 0.3,
        "patience": 8,
        "label_smoothing": 0.0,
    },
    {
        "name": "cfg_b",
        "emb_dim": 200,
        "hidden_dim": 256,
        "num_layers": 2,
        "max_len": 64,
        "epochs": 40,
        "batch_size": 128,
        "lr": 7e-4,
        "dropout": 0.4,
        "patience": 8,
        "label_smoothing": 0.05,
    },
    {
        "name": "cfg_c",
        "emb_dim": 128,
        "hidden_dim": 128,
        "num_layers": 1,
        "max_len": 48,
        "epochs": 40,
        "batch_size": 64,
        "lr": 2e-3,
        "dropout": 0.2,
        "patience": 6,
        "label_smoothing": 0.0,
    },
]
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


# ── Vocabulary & tokenization ─────────────────────────────────────────────────
# Keep emojis/punctuation by splitting on whitespace.
def tokenise(text: str):
    return re.findall(r"\S+", str(text).lower())


def build_vocab(texts, vocab_size: int) -> dict:
    counter = Counter(tok for t in texts for tok in tokenise(t))
    tokens  = [w for w, _ in counter.most_common(vocab_size - 2)]
    vocab   = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(tokens)})
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(t, 1) for t in tokenise(text)[:max_len)]
    ids += [0] * (max_len - len(ids))
    return ids


# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: dict, max_len: int):
        self.data   = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels  # may be None for test set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx])
        y = int(self.labels[idx]) if self.labels is not None else -1
        return x, y


# ── Attention Layer ───────────────────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, rnn_outputs, mask):
        attn_scores = self.attn(rnn_outputs).squeeze(-1)  # (B, L)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L)
        context = torch.sum(rnn_outputs * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


# ── Model ─────────────────────────────────────────────────────────────────────
class BiGRU(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.rnn  = nn.GRU(
            emb_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.drop(self.emb(x))
        rnn_outputs, _ = self.rnn(emb)  # (B, L, 2H)
        mask = x != 0
        context, _ = self.attention(rnn_outputs, mask)
        return self.fc(self.drop(context))


# ── Training helpers ──────────────────────────────────────────────────────────
def compute_class_weights(labels, num_classes: int):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            loss = criterion(logits, y.to(device))
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    avg_loss = total_loss / len(loader)
    return f1_score(all_labels, all_preds, average="macro"), avg_loss


# ── Visualization Functions ────────────────────────────────────────────────
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_training_curves(train_losses, val_losses, val_f1s):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # F1 Score plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1s, label='Validation Macro-F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_attention(sample_text, vocab, model):
    model.eval()
    tokens = tokenise(sample_text)
    encoded = torch.LongTensor([encode(sample_text, vocab, MAX_LEN)]).to(DEVICE)
    with torch.no_grad():
        emb = model.emb(encoded)
        rnn_outputs, _ = model.rnn(emb)
        mask = encoded != 0
        _, attn_weights = model.attention(rnn_outputs, mask)

    attn_weights = attn_weights.squeeze(0).cpu().numpy()
    plt.figure(figsize=(12, 2))
    sns.heatmap([attn_weights[:len(tokens)]], annot=[tokens], fmt='', cmap='Reds', cbar=False)
    plt.title('Attention Weights')
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv("cleaned_data/train_cleaned.csv")
    valid_df = pd.read_csv("cleaned_data/valid_cleaned.csv")
    test_df  = pd.read_csv("cleaned_data/test_cleaned.csv")

    print("Building vocabulary …")
    vocab = build_vocab(train_df["cleaned_text"], VOCAB_SIZE)

    train_ds = TextDataset(train_df["cleaned_text"], train_df["label"].values, vocab, MAX_LEN)
    valid_ds = TextDataset(valid_df["cleaned_text"], valid_df["label"].values, vocab, MAX_LEN)
    test_ds  = TextDataset(test_df["cleaned_text"],  None,                     vocab, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=256,        shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=0)

    model     = BiGRU(len(vocab), EMB_DIM, HIDDEN_DIM, NUM_LAYERS,
                      NUM_CLASSES, DROPOUT).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
    )

    class_weights = compute_class_weights(train_df["label"].values, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    best_f1, best_state = 0.0, None
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_f1, val_loss = evaluate(model, valid_loader, DEVICE, criterion)
        scheduler.step(val_f1)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"loss={loss:.4f} | val_loss={val_loss:.4f} | val Macro-F1={val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nBest Validation Macro-F1: {best_f1:.4f}")

    # ── Generate test predictions ────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            preds = model(X.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    out = pd.DataFrame({"id": test_df["id"], "label": all_preds})
    os.makedirs("predictions", exist_ok=True)
    out.to_csv("predictions/bigru_attn_pred_"+str(best_f1)+".csv", index=False)
    print("Saved bigru_attn_pred.csv")


if __name__ == "__main__":
    main()
