"""
Trains Models 1-5 from the notebook and saves lightweight artifacts
(vectorizers, classifiers, NN weights) so the Streamlit app can load
them instantly instead of retraining on every deploy/restart.

Run once: python train_artifacts.py
Outputs into ./artifacts/
"""
import csv
import json
import math
import os
import pickle
import random
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

LABELS = ["A", "B", "C", "D", "E"]
ART_DIR = os.path.join(os.path.dirname(__file__), "app", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


train_rows_all = load_csv(os.path.join(os.path.dirname(__file__), "app", "train.csv"))
train_rows, val_rows = train_test_split(
    train_rows_all, test_size=0.15, random_state=SEED,
    stratify=[r["answer"] for r in train_rows_all],
)
print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")


def map_at_3(preds, truth):
    scores = []
    for p, t in zip(preds, truth):
        scores.append(next((1 / (i + 1) for i, x in enumerate(p[:3]) if x == t), 0.0))
    return float(np.mean(scores))


def top1_acc(preds, truth):
    return sum(p[0] == t for p, t in zip(preds, truth)) / len(truth)


# ── Model 1: TF-IDF Cosine Ranker ────────────────────────────────────────────
class TFIDFRanker:
    def __init__(self, max_features=20_000, min_df=2):
        self.vec = TfidfVectorizer(
            max_features=max_features, min_df=min_df, ngram_range=(1, 2),
            sublinear_tf=True, stop_words="english",
        )

    def fit(self, rows):
        corpus = [r["prompt"] + " " + " ".join(r[l] for l in LABELS) for r in rows]
        self.vec.fit(corpus)
        return self

    def predict_single(self, row):
        vecs = self.vec.transform([row["prompt"]] + [row[l] for l in LABELS])
        sims = cosine_similarity(vecs[0], vecs[1:])[0]
        return [LABELS[i] for i in sims.argsort()[::-1]], sims

    def predict(self, rows):
        return [self.predict_single(r)[0] for r in rows]


print("Training Model 1 (TF-IDF Cosine)...")
model1 = TFIDFRanker().fit(train_rows)
p1 = model1.predict(val_rows)
truth = [r["answer"] for r in val_rows]
print(f"  Model1 MAP@3={map_at_3(p1, truth):.4f} Top1={top1_acc(p1, truth):.4f}")
with open(os.path.join(ART_DIR, "model1_tfidf_ranker.pkl"), "wb") as f:
    pickle.dump(model1, f)


# ── Model 2: TF-IDF + Logistic Regression ────────────────────────────────────
class SklearnMCQClassifier:
    def __init__(self, max_features=30_000, C=1.0):
        self.vec = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2),
            sublinear_tf=True, strip_accents="unicode",
        )
        self.clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", n_jobs=-1)

    def _build(self, rows, fit=False):
        texts, y = [], []
        for row in rows:
            for lbl in LABELS:
                texts.append(row["prompt"] + " [SEP] " + row[lbl])
                if "answer" in row:
                    y.append(1 if lbl == row["answer"] else 0)
        X = self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        return X, (np.array(y) if y else None)

    def fit(self, rows):
        X, y = self._build(rows, fit=True)
        self.clf.fit(X, y)
        return self

    def predict_proba_ranked(self, rows):
        X, _ = self._build(rows)
        probs = self.clf.predict_proba(X)[:, 1].reshape(len(rows), 5)
        ranked = [[LABELS[j] for j in np.argsort(-probs[i])] for i in range(len(rows))]
        return ranked, probs

    def predict(self, rows):
        return self.predict_proba_ranked(rows)[0]


print("Training Model 2 (TF-IDF + LogReg)...")
model2 = SklearnMCQClassifier(max_features=30_000, C=1.0).fit(train_rows)
p2 = model2.predict(val_rows)
print(f"  Model2 MAP@3={map_at_3(p2, truth):.4f} Top1={top1_acc(p2, truth):.4f}")
with open(os.path.join(ART_DIR, "model2_tfidf_logreg.pkl"), "wb") as f:
    pickle.dump(model2, f)


# ── Model 3: Retrieval Ranker with LogReg fallback ───────────────────────────
class RetrievalRanker:
    def __init__(self):
        self.index: Dict[str, Dict] = {}
        self.fallback: Optional[SklearnMCQClassifier] = None

    def fit(self, rows):
        for row in rows:
            self.index[row["prompt"].strip()] = row
        self.fallback = SklearnMCQClassifier(max_features=30_000, C=1.0).fit(rows)
        return self

    def predict_single(self, row):
        key = row["prompt"].strip()
        if key in self.index:
            correct = self.index[key]["answer"]
            rest = [l for l in LABELS if l != correct]
            return [correct] + rest, True
        else:
            return self.fallback.predict([row])[0], False

    def predict(self, rows):
        return [self.predict_single(r)[0] for r in rows]


print("Training Model 3 (Retrieval + LogReg fallback)...")
model3 = RetrievalRanker().fit(train_rows_all)  # full data, as in notebook
p3 = model3.predict(val_rows)
print(f"  Model3 MAP@3={map_at_3(p3, truth):.4f} Top1={top1_acc(p3, truth):.4f}")
with open(os.path.join(ART_DIR, "model3_retrieval.pkl"), "wb") as f:
    pickle.dump(model3, f)


# ── Model 4: PyTorch MLP Scorer over TF-IDF features ─────────────────────────
NN_VOCAB = 20_000  # trimmed down from notebook's 30k for a lighter artifact


def make_pair_texts_rows(rows):
    texts = []
    for r in rows:
        for lbl in LABELS:
            texts.append(r["prompt"] + " [SEP] " + str(r[lbl]))
    return texts


print("Fitting TF-IDF for Model 4...")
nn_vec = TfidfVectorizer(max_features=NN_VOCAB, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
nn_vec.fit(make_pair_texts_rows(train_rows_all))


def get_nn_features(rows, vec):
    texts = make_pair_texts_rows(rows)
    X = vec.transform(texts).toarray().astype(np.float32)
    return torch.tensor(X.reshape(len(rows), 5, -1))


X_tr_nn = get_nn_features(train_rows, nn_vec)
X_val_nn = get_nn_features(val_rows, nn_vec)
label2idx = {l: i for i, l in enumerate(LABELS)}
y_tr_nn = torch.tensor([label2idx[r["answer"]] for r in train_rows], dtype=torch.long)
y_val_nn = torch.tensor([label2idx[r["answer"]] for r in val_rows], dtype=torch.long)
V = X_tr_nn.shape[-1]


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


print("Training Model 4 (PyTorch MLP Scorer)...")
device = torch.device("cpu")
model4 = MCQScorer(input_dim=V).to(device)
opt = torch.optim.AdamW(model4.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
crit = nn.CrossEntropyLoss()

from torch.utils.data import DataLoader, Dataset


class MCQDataset(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


train_dl4 = DataLoader(MCQDataset(X_tr_nn, y_tr_nn), batch_size=64, shuffle=True)
best_map4, best_state4 = 0.0, None

for ep in range(25):
    model4.train()
    for xb, yb in train_dl4:
        opt.zero_grad()
        loss = crit(model4(xb), yb)
        loss.backward()
        opt.step()
    sched.step()
    model4.eval()
    with torch.no_grad():
        logits = model4(X_val_nn)
        ranked = logits.argsort(dim=-1, descending=True).numpy()
        preds = [[LABELS[i] for i in row] for row in ranked]
    m = map_at_3(preds, truth)
    if m > best_map4:
        best_map4, best_state4 = m, {k: v.clone() for k, v in model4.state_dict().items()}

model4.load_state_dict(best_state4)
print(f"  Model4 best val MAP@3={best_map4:.4f}")
torch.save(model4.state_dict(), os.path.join(ART_DIR, "model4_mlp_scorer.pt"))
with open(os.path.join(ART_DIR, "model4_vectorizer.pkl"), "wb") as f:
    pickle.dump({"vec": nn_vec, "input_dim": V}, f)


# ── Model 5: Mini Transformer (scratch) ──────────────────────────────────────
def tok(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_vocab(rows, min_freq=2, max_vocab=12_000):
    c = Counter()
    for r in rows:
        c.update(tok(r["prompt"]))
        for l in LABELS:
            c.update(tok(str(r[l])))
    v = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
    for w, f in c.most_common(max_vocab):
        if f >= min_freq:
            v[w] = len(v)
    return v


print("Building vocab for Model 5...")
vocab5 = build_vocab(train_rows_all)
VOCAB5 = len(vocab5)
MAX_LEN5 = 96


def encode5(prompt, option, max_len=MAX_LEN5):
    p = [vocab5.get(w, 1) for w in tok(prompt)]
    o = [vocab5.get(w, 1) for w in tok(str(option))]
    ids = [vocab5["<CLS>"]] + p + [vocab5["<SEP>"]] + o
    ids = ids[:max_len]
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    return ids + [0] * (max_len - len(ids)), mask


class TransformerMCQDataset(Dataset):
    def __init__(self, rows, has_label=True):
        l2i = {l: i for i, l in enumerate(LABELS)}
        self.data = []
        for r in rows:
            idss, masks = [], []
            for l in LABELS:
                ids, mask = encode5(r["prompt"], r[l])
                idss.append(ids)
                masks.append(mask)
            self.data.append((
                torch.tensor(idss, dtype=torch.long),
                torch.tensor(masks, dtype=torch.float),
                l2i[r["answer"]] if has_label else -1,
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class PositionalEncoding(nn.Module):
    def __init__(self, d, L=MAX_LEN5, drop=0.1):
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
    def __init__(self, vocab_size, d_model=128, nhead=4, layers=2, dim_ff=256, max_len=MAX_LEN5, drop=0.1):
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
        # mean-pool over real tokens
        mask_exp = mask_f.unsqueeze(-1)
        pooled = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
        scores = self.scorer(pooled).view(B, N)
        return scores


print("Training Model 5 (Mini Transformer)...")
tr_ds5 = TransformerMCQDataset(train_rows)
val_ds5 = TransformerMCQDataset(val_rows)
tr_dl5 = DataLoader(tr_ds5, batch_size=32, shuffle=True)

model5 = MiniTransformer(VOCAB5)
opt5 = torch.optim.AdamW(model5.parameters(), lr=5e-4, weight_decay=1e-4)
sch5 = torch.optim.lr_scheduler.CosineAnnealingLR(opt5, T_max=15)
crit5 = nn.CrossEntropyLoss()


def predict5(model, ds, bs=32):
    model.eval()
    dl = DataLoader(ds, batch_size=bs)
    preds = []
    with torch.no_grad():
        for ids, mask, _ in dl:
            logits = model(ids, mask)
            ranked = logits.argsort(dim=-1, descending=True).numpy()
            for row in ranked:
                preds.append([LABELS[i] for i in row])
    return preds


best5, best_st5 = 0.0, None
for ep in range(15):
    model5.train()
    for ids, mask, y in tr_dl5:
        opt5.zero_grad()
        loss = crit5(model5(ids, mask), y)
        loss.backward()
        opt5.step()
    sch5.step()
    preds5 = predict5(model5, val_ds5)
    m = map_at_3(preds5, truth)
    if m > best5:
        best5, best_st5 = m, {k: v.clone() for k, v in model5.state_dict().items()}
    print(f"  ep{ep+1:02d} val MAP@3={m:.4f}")

model5.load_state_dict(best_st5)
print(f"  Model5 best val MAP@3={best5:.4f}")
torch.save(model5.state_dict(), os.path.join(ART_DIR, "model5_mini_transformer.pt"))
with open(os.path.join(ART_DIR, "model5_vocab.json"), "w") as f:
    json.dump({"vocab": vocab5, "max_len": MAX_LEN5, "vocab_size": VOCAB5}, f)

print("\nAll artifacts saved to", ART_DIR)
print(os.listdir(ART_DIR))
