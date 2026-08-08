"""
Smart MCQ Solver — Streamlit deployment for all 8 models from the notebook.

Run locally:
    streamlit run app.py
"""
import json
import math
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="wide")

LABELS = ["A", "B", "C", "D", "E"]
BASE_DIR = os.path.dirname(__file__)
ART_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_CATALOG = {
    "Model 1 — TF-IDF Cosine Similarity": "m1",
    "Model 2 — TF-IDF + Logistic Regression": "m2",
    "Model 3 — Retrieval + LogReg Fallback": "m3",
    "Model 4 — PyTorch MLP Scorer": "m4",
    "Model 5 — Mini Transformer (scratch)": "m5",
    "Model 6 — Sentence-BERT (MiniLM-L6-v2, zero-shot)": "m6",
    "Model 8 — Sentence-BERT (BGE-base, zero-shot)": "m8",
    "Model 7 — RAG + LLM (needs your OpenAI API key)": "m7",
}

LIGHTWEIGHT = {"m1", "m2", "m3", "m4", "m5"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def map_at_3(preds, truth):
    scores = []
    for p, t in zip(preds, truth):
        scores.append(next((1 / (i + 1) for i, x in enumerate(p[:3]) if x == t), 0.0))
    return float(np.mean(scores))


def top1_acc(preds, truth):
    return sum(p[0] == t for p, t in zip(preds, truth)) / len(truth)


def row_from_inputs(prompt, opts):
    row = {"prompt": prompt}
    row.update({l: o for l, o in zip(LABELS, opts)})
    return row


def scores_dataframe(labels_ranked, scores):
    order = {l: s for l, s in zip(labels_ranked, scores)}
    df = pd.DataFrame({"Option": LABELS, "Score": [order.get(l, 0.0) for l in LABELS]})
    return df.sort_values("Score", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pickle-compat class definitions (must match train_artifacts.py exactly so
# pickle can resolve them when unpickling model1/2/3 artifacts)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — TF-IDF Cosine Ranker
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Model 1 (TF-IDF Cosine)...")
def load_model1():
    with open(os.path.join(ART_DIR, "model1_tfidf_ranker.pkl"), "rb") as f:
        return pickle.load(f)


def predict_model1(model, row):
    from sklearn.metrics.pairwise import cosine_similarity

    vecs = model.vec.transform([row["prompt"]] + [row[l] for l in LABELS])
    sims = cosine_similarity(vecs[0], vecs[1:])[0]
    ranked = [LABELS[i] for i in sims.argsort()[::-1]]
    return ranked, sims[np.argsort(-sims)]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Model 2 (TF-IDF + LogReg)...")
def load_model2():
    with open(os.path.join(ART_DIR, "model2_tfidf_logreg.pkl"), "rb") as f:
        return pickle.load(f)


def predict_model2(model, row):
    texts = [row["prompt"] + " [SEP] " + row[l] for l in LABELS]
    X = model.vec.transform(texts)
    probs = model.clf.predict_proba(X)[:, 1]
    order = np.argsort(-probs)
    ranked = [LABELS[i] for i in order]
    return ranked, probs[order]


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Retrieval + LogReg fallback
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Model 3 (Retrieval Ranker)...")
def load_model3():
    with open(os.path.join(ART_DIR, "model3_retrieval.pkl"), "rb") as f:
        return pickle.load(f)


def predict_model3(model, row):
    key = row["prompt"].strip()
    if key in model.index:
        correct = model.index[key]["answer"]
        rest = [l for l in LABELS if l != correct]
        ranked = [correct] + rest
        scores = [1.0] + [0.0] * 4
        return ranked, np.array(scores), True
    else:
        ranked, scores = predict_model2(model.fallback, row)
        return ranked, scores, False


# ─────────────────────────────────────────────────────────────────────────────
# Model 4 — PyTorch MLP Scorer
# ─────────────────────────────────────────────────────────────────────────────
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


@st.cache_resource(show_spinner="Loading Model 4 (PyTorch MLP Scorer)...")
def load_model4():
    with open(os.path.join(ART_DIR, "model4_vectorizer.pkl"), "rb") as f:
        meta = pickle.load(f)
    model = MCQScorer(input_dim=meta["input_dim"])
    state = torch.load(os.path.join(ART_DIR, "model4_mlp_scorer.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta["vec"]


def predict_model4(model_and_vec, row):
    model, vec = model_and_vec
    texts = [row["prompt"] + " [SEP] " + str(row[l]) for l in LABELS]
    X = vec.transform(texts).toarray().astype(np.float32)
    X = torch.tensor(X).unsqueeze(0)  # (1, 5, V)
    with torch.no_grad():
        logits = model(X)[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    order = np.argsort(-probs)
    ranked = [LABELS[i] for i in order]
    return ranked, probs[order]


# ─────────────────────────────────────────────────────────────────────────────
# Model 5 — Mini Transformer (scratch)
# ─────────────────────────────────────────────────────────────────────────────
def tok(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


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


@st.cache_resource(show_spinner="Loading Model 5 (Mini Transformer)...")
def load_model5():
    with open(os.path.join(ART_DIR, "model5_vocab.json")) as f:
        meta = json.load(f)
    model = MiniTransformer(meta["vocab_size"], max_len=meta["max_len"])
    state = torch.load(os.path.join(ART_DIR, "model5_mini_transformer.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def encode5(prompt, option, vocab, max_len):
    p = [vocab.get(w, 1) for w in tok(prompt)]
    o = [vocab.get(w, 1) for w in tok(str(option))]
    ids = [vocab["<CLS>"]] + p + [vocab["<SEP>"]] + o
    ids = ids[:max_len]
    mask = [1] * len(ids) + [0] * (max_len - len(ids))
    return ids + [0] * (max_len - len(ids)), mask


def predict_model5(model_and_meta, row):
    model, meta = model_and_meta
    vocab, max_len = meta["vocab"], meta["max_len"]
    idss, masks = [], []
    for l in LABELS:
        ids, mask = encode5(row["prompt"], row[l], vocab, max_len)
        idss.append(ids)
        masks.append(mask)
    ids_t = torch.tensor([idss], dtype=torch.long)
    mask_t = torch.tensor([masks], dtype=torch.float)
    with torch.no_grad():
        logits = model(ids_t, mask_t)[0]
    probs = torch.softmax(logits, dim=-1).numpy()
    order = np.argsort(-probs)
    ranked = [LABELS[i] for i in order]
    return ranked, probs[order]


# ─────────────────────────────────────────────────────────────────────────────
# Models 6 / 8 — Sentence-BERT zero-shot (lazy, heavier download)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Downloading & loading sentence-transformer (first run only)...")
def load_sbert(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def predict_sbert(model, row):
    prompt_emb = model.encode([row["prompt"]], normalize_embeddings=True, convert_to_numpy=True)[0]
    opt_embs = model.encode([row[l] for l in LABELS], normalize_embeddings=True, convert_to_numpy=True)
    sims = opt_embs @ prompt_emb
    order = np.argsort(-sims)
    ranked = [LABELS[i] for i in order]
    return ranked, sims[order]


# ─────────────────────────────────────────────────────────────────────────────
# Model 7 — RAG + LLM (user supplies their own OpenAI key)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building TF-IDF retrieval index for RAG...")
def load_rag_retriever():
    train_path = os.path.join(BASE_DIR, "train.csv")
    df = pd.read_csv(train_path)
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = df["prompt"].fillna("").tolist()
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(corpus)
    return vec, X, df


def rag_retrieve(vec, X, df, query, k=5):
    from sklearn.metrics.pairwise import cosine_similarity

    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    top_idx = sims.argsort()[::-1][:k]
    return df.iloc[top_idx]


def predict_model7(api_key, row):
    from openai import OpenAI

    vec, X, df = load_rag_retriever()
    examples_df = rag_retrieve(vec, X, df, row["prompt"], k=5)

    ex_lines = []
    for _, r in examples_df.iterrows():
        opts = "\n".join(f"  {l}: {r[l]}" for l in LABELS)
        ex_lines.append(f"Q: {r['prompt']}\n{opts}\nAnswer: {r['answer']}")
    examples = "\n\n".join(ex_lines)

    choices = "\n".join(f"{l}: {row[l]}" for l in LABELS)
    prompt = (
        "You are an expert at answering multiple-choice questions.\n\n"
        "Here are some example questions with their correct answers:\n\n"
        f"{examples}\n"
        "────────────────────────────────────────────────\n"
        "Now answer this question. Reply with ONLY the single letter (A/B/C/D/E).\n\n"
        f"Question: {row['prompt']}\n{choices}\n\nAnswer:"
    )

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0,
    )
    text = resp.choices[0].message.content.strip().upper()
    top = next((c for c in text if c in LABELS), "A")
    rest = [l for l in LABELS if l != top]
    return [top] + rest, None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Smart MCQ Solver")
st.sidebar.caption("Deployed from the 8-model comparison notebook")

model_label = st.sidebar.selectbox("Choose a model", list(MODEL_CATALOG.keys()), index=1)
model_key = MODEL_CATALOG[model_label]

openai_key = None
if model_key == "m7":
    openai_key = st.sidebar.text_input(
        "Your OpenAI API key", type="password",
        help="Used only for this session, never stored or logged.",
    )
    st.sidebar.caption("⚠️ Model 7 calls the OpenAI API and needs your own key — the notebook's hardcoded key has been removed.")

if model_key in ("m6", "m8"):
    st.sidebar.caption("⚠️ First use downloads a pretrained model from Hugging Face (~90–450MB) and may take a minute.")

with st.sidebar.expander("ℹ️ About the models"):
    st.markdown(
        """
- **1–5**: trained from scratch on this competition's data, weights bundled with the app (instant).
- **6, 8**: pretrained sentence embedding models used *zero-shot* (cosine similarity between question and option embeddings) — no fine-tuning is run in this app for speed.
- **7**: retrieval-augmented generation — retrieves similar solved questions via TF-IDF, then asks an LLM. Requires your own OpenAI API key.
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_compare = st.tabs(["🔮 Try a question", "📄 Batch predict (CSV)", "📊 Model comparison"])

# ---------------- Tab 1: single prediction ----------------
with tab_predict:
    st.subheader("Answer a multiple-choice question")
    col1, col2 = st.columns([3, 2])
    with col1:
        prompt = st.text_area("Question prompt", height=100, placeholder="e.g. What is the capital of France?")
        opt_cols = st.columns(1)
        opts = []
        for l in LABELS:
            opts.append(st.text_input(f"Option {l}", key=f"opt_{l}"))

    with col2:
        st.markdown("#### Prediction")
        run = st.button("Predict answer", type="primary", use_container_width=True)

        if run:
            if not prompt.strip() or any(not o.strip() for o in opts):
                st.warning("Please fill in the prompt and all 5 options.")
            else:
                row = row_from_inputs(prompt, opts)
                try:
                    if model_key == "m1":
                        model = load_model1()
                        ranked, scores = predict_model1(model, row)
                    elif model_key == "m2":
                        model = load_model2()
                        ranked, scores = predict_model2(model, row)
                    elif model_key == "m3":
                        model = load_model3()
                        ranked, scores, hit = predict_model3(model, row)
                        if hit:
                            st.info("Exact match found in the training index — answer retrieved directly.")
                    elif model_key == "m4":
                        model = load_model4()
                        ranked, scores = predict_model4(model, row)
                    elif model_key == "m5":
                        model = load_model5()
                        ranked, scores = predict_model5(model, row)
                    elif model_key == "m6":
                        model = load_sbert("all-MiniLM-L6-v2")
                        ranked, scores = predict_sbert(model, row)
                    elif model_key == "m8":
                        model = load_sbert("BAAI/bge-base-en-v1.5")
                        ranked, scores = predict_sbert(model, row)
                    elif model_key == "m7":
                        if not openai_key:
                            st.error("Enter your OpenAI API key in the sidebar first.")
                            st.stop()
                        ranked, scores = predict_model7(openai_key, row)

                    st.success(f"**Predicted answer: {ranked[0]}**")
                    st.caption(f"Top-3 ranked: {', '.join(ranked[:3])}")

                    if scores is not None:
                        df = pd.DataFrame({"Option": ranked, "Score": scores})
                        st.bar_chart(df.set_index("Option"))
                except FileNotFoundError:
                    st.error(
                        "Model artifacts not found. Run `python train_artifacts.py` "
                        "once before starting the app (see README)."
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---------------- Tab 2: batch CSV ----------------
with tab_batch:
    st.subheader("Batch-predict from a CSV")
    st.caption("CSV needs columns: prompt, A, B, C, D, E (an optional `answer` column enables scoring).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    max_rows = st.slider("Max rows to process (for speed)", 10, 500, 100, step=10)

    if up is not None:
        df_in = pd.read_csv(up).head(max_rows)
        st.write(f"Loaded {len(df_in)} rows.")
        if st.button("Run batch prediction"):
            rows = df_in.to_dict("records")
            preds = []
            progress = st.progress(0.0)

            # Load the chosen model once
            if model_key == "m1":
                model = load_model1()
                fn = lambda r: predict_model1(model, r)
            elif model_key == "m2":
                model = load_model2()
                fn = lambda r: predict_model2(model, r)
            elif model_key == "m3":
                model = load_model3()
                fn = lambda r: predict_model3(model, r)[:2]
            elif model_key == "m4":
                model = load_model4()
                fn = lambda r: predict_model4(model, r)
            elif model_key == "m5":
                model = load_model5()
                fn = lambda r: predict_model5(model, r)
            elif model_key == "m6":
                model = load_sbert("all-MiniLM-L6-v2")
                fn = lambda r: predict_sbert(model, r)
            elif model_key == "m8":
                model = load_sbert("BAAI/bge-base-en-v1.5")
                fn = lambda r: predict_sbert(model, r)
            elif model_key == "m7":
                if not openai_key:
                    st.error("Enter your OpenAI API key in the sidebar first.")
                    st.stop()
                fn = lambda r: predict_model7(openai_key, r)

            for i, r in enumerate(rows):
                ranked, _ = fn(r)
                preds.append(ranked)
                progress.progress((i + 1) / len(rows))

            out = pd.DataFrame({
                "id": df_in["id"] if "id" in df_in.columns else range(len(df_in)),
                "Prediction": [" ".join(p[:3]) for p in preds],
                "top1": [p[0] for p in preds],
            })
            st.dataframe(out, use_container_width=True)

            if "answer" in df_in.columns:
                truth = df_in["answer"].tolist()
                st.metric("MAP@3", f"{map_at_3(preds, truth):.4f}")
                st.metric("Top-1 accuracy", f"{top1_acc(preds, truth):.4f}")

            csv_bytes = out.to_csv(index=False).encode()
            st.download_button("⬇️ Download submission.csv", csv_bytes, "submission.csv", "text/csv")

# ---------------- Tab 3: model comparison ----------------
with tab_compare:
    st.subheader("Validation-set comparison (Models 1–5)")
    metrics_path = os.path.join(ART_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        mdf = pd.DataFrame(metrics).T
        mdf.index.name = "Model"
        st.dataframe(mdf.style.highlight_max(axis=0, color="#d4f5dd"), use_container_width=True)
        st.bar_chart(mdf[["map@3"]])
    else:
        st.info("Run `train_artifacts.py` to generate `artifacts/metrics.json` for this comparison table.")
    st.caption(
        "Models 6, 7 and 8 aren't included in this table because they either need a live "
        "API key (Model 7) or a downloaded pretrained model, so their scores vary by run/config."
    )
