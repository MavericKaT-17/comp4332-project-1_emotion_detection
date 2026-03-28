import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from collections import Counter

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
VOCAB_SIZE  = 30_000
EMB_DIM     = 100
MAX_LEN     = 64
EPOCHS      = 20
BATCH_SIZE  = 256
LR          = 1e-3
DROPOUT     = 0.3
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


# ── Vocabulary & tokenization ─────────────────────────────────────────────────
def tokenise(text: str):
    return re.findall(r"\w+", str(text).lower())


def build_vocab(texts, vocab_size: int) -> dict:
    counter = Counter(tok for t in texts for tok in tokenise(t))
    tokens  = [w for w, _ in counter.most_common(vocab_size - 2)]
    vocab   = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(tokens)})
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(t, 1) for t in tokenise(text)[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: dict, max_len: int):
        self.data   = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx])
        y = int(self.labels[idx]) if self.labels is not None else -1
        return x, y


# ── Model ─────────────────────────────────────────────────────────────────────
class EmbMLP(nn.Module):
    """Mean-pool word embeddings → 3-layer MLP classifier."""

    def __init__(self, vocab_size: int, emb_dim: int, num_classes: int,
                 dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, L)  — 0 is <PAD>
        mask   = (x != 0).unsqueeze(-1).float()          # (B, L, 1)
        emb    = self.emb(x)                             # (B, L, E)
        summed = (emb * mask).sum(dim=1)                 # (B, E)
        counts = mask.sum(dim=1).clamp(min=1)            # (B, 1)
        pooled = summed / counts                         # mean over non-PAD tokens
        return self.mlp(pooled)


# ── Helpers ───────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return f1_score(all_labels, all_preds, average="macro")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    test_df  = pd.read_csv("data/test_no_label.csv")

    print("Building vocabulary …")
    vocab = build_vocab(train_df["text"], VOCAB_SIZE)

    train_ds = TextDataset(train_df["text"], train_df["label"].values, vocab, MAX_LEN)
    valid_ds = TextDataset(valid_df["text"], valid_df["label"].values, vocab, MAX_LEN)
    test_ds  = TextDataset(test_df["text"],  None,                     vocab, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=512,        shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False, num_workers=0)

    model     = EmbMLP(len(vocab), EMB_DIM, NUM_CLASSES, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        loss   = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_f1 = evaluate(model, valid_loader, DEVICE)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"loss={loss:.4f} | val Macro-F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

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
    out.to_csv("emb_mlp_pred.csv", index=False)
    print("Saved emb_mlp_pred.csv")


if __name__ == "__main__":
    main()
