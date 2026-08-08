"""Recomputes validation metrics for Models 1-5 from saved artifacts and
writes app/artifacts/metrics.json for the Streamlit 'Model comparison' tab."""
import csv
import json
import math
import os
import pickle
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

LABELS = ["A", "B", "C", "D", "E"]
BASE = os.path.dirname(__file__)
ART_DIR = os.path.join(BASE, "app", "artifacts")
SEED = 42


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


train_rows_all = load_csv(os.path.join(BASE, "app", "train.csv"))
train_rows, val_rows = train_test_split(
    train_rows_all, test_size=0.15, random_state=SEED,
    stratify=[r["answer"] for r in train_rows_all],
)
truth = [r["answer"] for r in val_rows]


def map_at_3(preds, truth):
    scores = []
    for p, t in zip(preds, truth):
        scores.append(next((1 / (i + 1) for i, x in enumerate(p[:3]) if x == t), 0.0))
    return float(np.mean(scores))


def top1_acc(preds, truth):
    return sum(p[0] == t for p, t in zip(preds, truth)) / len(truth)


def top3_acc(preds, truth):
    return sum(t in p[:3] for p, t in zip(preds, truth)) / len(truth)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Classes must be redefined here (identically) so pickle can resolve them
# under __main__ when unpickling.
class TFIDFRanker:
    def __init__(self, max_features=20_000, min_df=2):
        self.vec = TfidfVectorizer(
            max_features=max_features, min_df=min_df, ngram_range=(1, 2),
            sublinear_tf=True, stop_words="english",
        )


class SklearnMCQClassifier:
    def __init__(self, max_features=30_000, C=1.0):
        self.vec = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            sublinear_tf=True, strip_accents="unicode",
        )
        self.clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", n_jobs=-1)


class RetrievalRanker:
    def __init__(self):
        self.index = {}
        self.fallback = None


metrics = {}

# Model 1
with open(os.path.join(ART_DIR, "model1_tfidf_ranker.pkl"), "rb") as f:
    m1 = pickle.load(f)
preds = []
for r in val_rows:
    vecs = m1.vec.transform([r["prompt"]] + [r[l] for l in LABELS])
    sims = cosine_similarity(vecs[0], vecs[1:])[0]
    preds.append([LABELS[i] for i in sims.argsort()[::-1]])
metrics["Model 1 - TF-IDF Cosine"] = {"map@3": map_at_3(preds, truth), "top1": top1_acc(preds, truth), "top3": top3_acc(preds, truth)}

# Model 2
with open(os.path.join(ART_DIR, "model2_tfidf_logreg.pkl"), "rb") as f:
    m2 = pickle.load(f)


def predict_m2(model, rows):
    texts = []
    for row in rows:
        for lbl in LABELS:
            texts.append(row["prompt"] + " [SEP] " + row[lbl])
    X = model.vec.transform(texts)
    probs = model.clf.predict_proba(X)[:, 1].reshape(len(rows), 5)
    return [[LABELS[j] for j in np.argsort(-probs[i])] for i in range(len(rows))]


preds2 = predict_m2(m2, val_rows)
metrics["Model 2 - TF-IDF + LogReg"] = {"map@3": map_at_3(preds2, truth), "top1": top1_acc(preds2, truth), "top3": top3_acc(preds2, truth)}

# Model 3
with open(os.path.join(ART_DIR, "model3_retrieval.pkl"), "rb") as f:
    m3 = pickle.load(f)
preds3 = []
for r in val_rows:
    key = r["prompt"].strip()
    if key in m3.index:
        correct = m3.index[key]["answer"]
        preds3.append([correct] + [l for l in LABELS if l != correct])
    else:
        preds3.append(predict_m2(m3.fallback, [r])[0])
metrics["Model 3 - Retrieval + Fallback"] = {"map@3": map_at_3(preds3, truth), "top1": top1_acc(preds3, truth), "top3": top3_acc(preds3, truth)}

# Model 4
with open(os.path.join(ART_DIR, "model4_vectorizer.pkl"), "rb") as f:
    meta4 = pickle.load(f)


class MCQScorer(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=128, hidden3=32, dropout=0.30):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.BatchNorm1d(hidden1), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.BatchNorm1d(hidden2), nn.Dropout(dropout * 0.7),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, 1),
        )

    def forward(self, x):
        B, N, V_ = x.shape
        s = self.scorer(x.view(B * N, V_))
        return s.view(B, N)


model4 = MCQScorer(input_dim=meta4["input_dim"])
model4.load_state_dict(torch.load(os.path.join(ART_DIR, "model4_mlp_scorer.pt"), map_location="cpu"))
model4.eval()


def make_pair_texts_rows(rows):
    texts = []
    for r in rows:
        for lbl in LABELS:
            texts.append(r["prompt"] + " [SEP] " + str(r[lbl]))
    return texts


X_val = meta4["vec"].transform(make_pair_texts_rows(val_rows)).toarray().astype(np.float32)
X_val = torch.tensor(X_val.reshape(len(val_rows), 5, -1))
with torch.no_grad():
    logits = model4(X_val)
    ranked = logits.argsort(dim=-1, descending=True).numpy()
preds4 = [[LABELS[i] for i in row] for row in ranked]
metrics["Model 4 - PyTorch MLP Scorer"] = {"map@3": map_at_3(preds4, truth), "top1": top1_acc(preds4, truth), "top3": top3_acc(preds4, truth)}

# Model 5
with open(os.path.join(ART_DIR, "model5_vocab.json")) as f:
    meta5 = json.load(f)
vocab5, max_len5 = meta5["vocab"], meta5["max_len"]


def tok(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def encode5(prompt, option):
    p = [vocab5.get(w, 1) for w in tok(prompt)]
    o = [vocab5.get(w, 1) for w in tok(str(option))]
    ids = [vocab5["<CLS>"]] + p + [vocab5["<SEP>"]] + o
    ids = ids[:max_len5]
    mask = [1] * len(ids) + [0] * (max_len5 - len(ids))
    return ids + [0] * (max_len5 - len(ids)), mask


class PositionalEncoding(nn.Module):
    def __init__(self, d, L, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, dim_ff=256, max_len=96, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len, drop)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.scorer = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1))
        self.d_model = d_model

    def forward(self, ids, mask):
        B, N, L = ids.shape
        ids_f = ids.view(B * N, L)
        mask_f = mask.view(B * N, L)
        x = self.emb(ids_f) * math.sqrt(self.d_model)
        x = self.pos(x)
        pad_mask = mask_f == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask_exp = mask_f.unsqueeze(-1)
        pooled = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        return self.scorer(pooled).view(B, N)


model5 = MiniTransformer(meta5["vocab_size"], max_len=max_len5)
model5.load_state_dict(torch.load(os.path.join(ART_DIR, "model5_mini_transformer.pt"), map_location="cpu"))
model5.eval()

idss_all, masks_all = [], []
for r in val_rows:
    idss, masks = [], []
    for l in LABELS:
        ids, mask = encode5(r["prompt"], r[l])
        idss.append(ids)
        masks.append(mask)
    idss_all.append(idss)
    masks_all.append(masks)
ids_t = torch.tensor(idss_all, dtype=torch.long)
mask_t = torch.tensor(masks_all, dtype=torch.float)
preds5 = []
with torch.no_grad():
    for i in range(0, len(val_rows), 32):
        logits = model5(ids_t[i:i+32], mask_t[i:i+32])
        ranked = logits.argsort(dim=-1, descending=True).numpy()
        preds5.extend([[LABELS[j] for j in row] for row in ranked])
metrics["Model 5 - Mini Transformer"] = {"map@3": map_at_3(preds5, truth), "top1": top1_acc(preds5, truth), "top3": top3_acc(preds5, truth)}

with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print("\nSaved to", os.path.join(ART_DIR, "metrics.json"))
