#!/usr/bin/env python
# coding: utf-8

# ## Step 0. Imports

# In[5]:


import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score


# ## Step 1. Load K-mer Processed Data

# In[6]:


DATA_DIR = "data_processed_kmer"

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))

Y_train = load_npz(os.path.join(DATA_DIR, "Y_train.npz"))
Y_val   = load_npz(os.path.join(DATA_DIR, "Y_val.npz"))

train_idx = np.load(os.path.join(DATA_DIR, "train_idx.npy"))
val_idx   = np.load(os.path.join(DATA_DIR, "val_idx.npy"))

with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

print("X_train:", X_train.shape, X_train.dtype)
print("X_val:  ", X_val.shape, X_val.dtype)
print("Y_train:", Y_train.shape, type(Y_train))
print("Y_val:  ", Y_val.shape, type(Y_val))
print("train_idx:", train_idx.shape)
print("val_idx:  ", val_idx.shape)
print("meta keys:", list(meta.keys()))


# In[7]:


print("Input dim (k-mer):", X_train.shape[1])
print("Output dim (all GO terms):", Y_train.shape[1])

if "mlb" in meta:
    mlb = meta["mlb"]
    print("Number of GO classes from mlb:", len(mlb.classes_))
    print("First 10 GO terms:", mlb.classes_[:10])

if "protein_ids" in meta:
    protein_ids = meta["protein_ids"]
    print("Number of protein_ids:", len(protein_ids))


# In[8]:


class ProteinKmerDataset(Dataset):
    def __init__(self, X, Y_sparse):
        self.X = X
        self.Y = Y_sparse

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx].toarray().ravel(), dtype=torch.float32)
        return x, y


# In[9]:


train_dataset = ProteinKmerDataset(X_train, Y_train)
val_dataset   = ProteinKmerDataset(X_val, Y_val)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Train batches:", len(train_loader))
print("Val batches:  ", len(val_loader))


# In[10]:


class KmerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=512, hidden2=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# In[11]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = KmerMLP(
    input_dim=X_train.shape[1],   # 400
    output_dim=Y_train.shape[1]   # 31454
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", f"{n_params:,}")
print(model)


# In[12]:


label_counts = np.asarray(Y_train.sum(axis=0)).ravel().astype(np.float32)
n_train = Y_train.shape[0]

# avoid being devided by 0
pos_weight = (n_train - label_counts) / (label_counts + 1e-6)

print("label_counts shape:", label_counts.shape)
print("pos_weight shape:", pos_weight.shape)
print("pos_weight min/max:", pos_weight.min(), pos_weight.max())

pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# In[13]:


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


# In[14]:


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(Y_batch.numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    return all_probs, all_targets


# In[15]:


def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)

    records = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int8)

        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        records.append({
            "threshold": float(t),
            "micro_f1": float(micro),
            "macro_f1": float(macro)
        })

        print(f"t={t:.2f} | micro-F1={micro:.4f} | macro-F1={macro:.4f}")

    best_micro_row = max(records, key=lambda x: x["micro_f1"])
    best_macro_row = max(records, key=lambda x: x["macro_f1"])

    return records, best_micro_row, best_macro_row


# In[ ]:


num_epochs = 5
save_path = "best_kmer_mlp_full_go.pt"

train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate_loss(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:02d}/{num_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"  -> saved best model to {save_path}")


# In[ ]:


plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="train")
plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("K-mer MLP training curve (full GO)")
plt.legend()
plt.show()


# In[ ]:


best_model = KmerMLP(
    input_dim=X_train.shape[1],
    output_dim=Y_train.shape[1]
).to(device)

best_model.load_state_dict(torch.load(save_path, map_location=device))

val_probs, val_targets = get_predictions(best_model, val_loader, device)

print("val_probs shape:", val_probs.shape)
print("val_targets shape:", val_targets.shape)


# In[ ]:


thresholds = np.arange(0.05, 0.55, 0.05)
records, best_micro_row, best_macro_row = threshold_sweep(val_targets, val_probs, thresholds)

print("\nBest by micro-F1:", best_micro_row)
print("Best by macro-F1:", best_macro_row)


# In[ ]:


results = {
    "model": "kmer_mlp_full_go",
    "input_dim": int(X_train.shape[1]),
    "output_dim": int(Y_train.shape[1]),
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "best_val_loss": float(best_val_loss),
    "best_micro": best_micro_row,
    "best_macro": best_macro_row,
    "train_losses": train_losses,
    "val_losses": val_losses,
}

with open("kmer_mlp_full_go_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved metrics to kmer_mlp_full_go_metrics.json")


# In[ ]:


import pandas as pd

df_thresh = pd.DataFrame(records)
display(df_thresh)


# In[ ]:


plt.figure(figsize=(6, 4))
plt.plot(df_thresh["threshold"], df_thresh["micro_f1"], marker='o', label="micro-F1")
plt.plot(df_thresh["threshold"], df_thresh["macro_f1"], marker='o', label="macro-F1")
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.title("Threshold sweep (full GO)")
plt.legend()
plt.show()

