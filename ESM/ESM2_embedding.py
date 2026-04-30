ㄧ#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 1. Import libraries and define file paths
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy.sparse import csr_matrix, save_npz

# Update this path if your CAFA-5 folder lives somewhere else
BASE_DIR = "cafa-5-protein-function-prediction"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")

train_fasta_path = os.path.join(TRAIN_DIR, "train_sequences.fasta")
train_terms_path = os.path.join(TRAIN_DIR, "train_terms.tsv")
train_taxonomy_path = os.path.join(TRAIN_DIR, "train_taxonomy.tsv")

for p in [train_fasta_path, train_terms_path, train_taxonomy_path]:
    print(p, "->", os.path.exists(p))
## 2. Load raw data
# Load sequences from FASTA
sequences_dict = {}
duplicate_id_count = 0
missing_id_count = 0
empty_seq_count = 0

for record in SeqIO.parse(train_fasta_path, "fasta"):
    seq_id = record.id
    seq = str(record.seq).strip()

    if not seq_id:
        missing_id_count += 1
        continue

    if seq_id in sequences_dict:
        duplicate_id_count += 1

    if len(seq) == 0:
        empty_seq_count += 1

    sequences_dict[seq_id] = seq

print("Raw sequences loaded:", len(sequences_dict))
print("Duplicate FASTA IDs:", duplicate_id_count)
print("Missing IDs:", missing_id_count)
print("Empty sequences:", empty_seq_count)

# Load labels and taxonomy
terms = pd.read_csv(train_terms_path, sep="\t")
taxonomy = pd.read_csv(train_taxonomy_path, sep="\t")

print("train_terms shape:", terms.shape)
print("train_taxonomy shape:", taxonomy.shape)
terms.head()
## 3. Check missing values
print("Missing values in train_terms:")
print(terms.isnull().sum())
print()

print("Missing values in train_taxonomy:")
print(taxonomy.isnull().sum())
# Remove missing values in the label table
terms = terms.dropna(subset=["EntryID", "term"]).copy()
print("train_terms shape after dropping missing EntryID/term rows:", terms.shape)
## 4. Remove invalid sequences

# We remove:
# - empty sequences
# - sequences containing non-standard amino acids

# Standard amino acids:
# `ACDEFGHIKLMNPQRSTVWY`
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

clean_sequences_dict = {}
removed_empty_ids = []
removed_invalid = []   # (protein_id, bad_chars)

for seq_id, seq in sequences_dict.items():
    seq = seq.strip()

    if len(seq) == 0:
        removed_empty_ids.append(seq_id)
        continue

    bad = set(seq) - valid_aas
    if bad:
        removed_invalid.append((seq_id, bad))
        continue

    clean_sequences_dict[seq_id] = seq

print("Removed empty sequences:", len(removed_empty_ids))
print("Removed invalid sequences:", len(removed_invalid))
print("Remaining clean sequences:", len(clean_sequences_dict))
# Show a few removed invalid sequences
removed_invalid[:10]
# Replace the raw dictionary with the cleaned one
sequences_dict = clean_sequences_dict
## 5. Basic sequence statistics
lengths = [len(seq) for seq in sequences_dict.values()]

print("Number of proteins:", len(sequences_dict))
print("Mean length:", np.mean(lengths))
print("Median length:", np.median(lengths))
print("Min length:", np.min(lengths))
print("Max length:", np.max(lengths))
print("Std length:", np.std(lengths))

## 6. Build label dictionary and align sequences with labels
labels_dict = terms.groupby("EntryID")["term"].apply(list)

sequence_ids = set(sequences_dict.keys())
label_ids = set(labels_dict.index)

print("Proteins with sequence only:", len(sequence_ids - label_ids))
print("Proteins with label only:", len(label_ids - sequence_ids))
print("Proteins with both:", len(sequence_ids & label_ids))
protein_ids = []
sequences = []
all_labels = []

for pid, seq in sequences_dict.items():
    if pid in labels_dict:
        protein_ids.append(pid)
        sequences.append(seq)
        all_labels.append(labels_dict[pid])

print("Aligned proteins retained:", len(sequences))
## 7. Multi-label encoding
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(all_labels).astype(np.int8)

print("Original Y shape:", Y.shape)
print("Number of GO terms:", len(mlb.classes_))
labels_per_protein = Y.sum(axis=1)


label_counts = Y.sum(axis=0)

used_go_terms = set()

for labels in all_labels:
    used_go_terms.update(labels)

print("Number of GO used:", len(used_go_terms))
used_go_terms = set()

for labels in all_labels:
    used_go_terms.update(labels)

print("Number of GO used:", len(used_go_terms))
terms_clean_go = terms[terms["term"].isin(used_go_terms)]
aspect_dict = terms_clean_go.set_index("term")["aspect"].to_dict()
print(list(used_go_terms)[:20])
print("Total GO after cleaning:", len(used_go_terms))


# In[2]:


#!pip install transformers torch


# In[3]:


import torch
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/esm2_t30_150M_UR50D"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("Device:", device)


# In[4]:


def get_esm_embedding(seq):
    inputs = tokenizer(
        seq,
        return_tensors="pt",
        truncation=False
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state  # (1, L, 320)
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1)

    embedding = (summed / counts).squeeze().cpu().numpy()

    return embedding


# In[ ]:


from tqdm import tqdm
import numpy as np
import sys
import os

print("Start embedding...", flush=True)

save_every = 1000
esm_embeddings = []

for i, seq in enumerate(tqdm(sequences, file=sys.stdout, ncols=100)):
    emb = get_esm_embedding(seq)
    esm_embeddings.append(emb)

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(sequences)}", flush=True)

    if (i + 1) % save_every == 0:
        chunk = np.array(esm_embeddings)
        np.save(f"X_esm_chunk_{i+1}.npy", chunk)
        print(f"Saved chunk ending at {i+1}", flush=True)
        esm_embeddings = []

if esm_embeddings:
    chunk = np.array(esm_embeddings)
    np.save(f"X_esm_chunk_final.npy", chunk)
    print("Saved final chunk", flush=True)


# In[ ]:


print(X_esm.shape)
print(Y.shape)


# In[ ]:


print(X_esm[0][:10])
print("sum =", X_esm[0].sum())
print("std =", X_esm[0].std())


# In[ ]:


import numpy as np

print(np.allclose(X_esm[0], X_esm[1]))
print(np.allclose(X_esm[1], X_esm[2]))


# In[ ]:


OUTPUT_DIR = "data_processed_esm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "esm_dataset.npz"),
    X=X_esm,
    Y=Y,
    ids=protein_ids
)




