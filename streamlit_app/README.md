# Smart MCQ Solver — Streamlit App

Deploys all 8 models from your notebook behind one Streamlit UI, with a
model picker in the sidebar. Models 1–5 are pretrained and bundled as
artifacts so the app starts instantly; Models 6–8 load pretrained/live
resources on demand.

## What's inside

```
app/
  app.py              ← the Streamlit app
  requirements.txt
  train.csv, test.csv ← your competition data
  artifacts/          ← pretrained weights for Models 1-5 + metrics.json
train_artifacts.py    ← script used to (re)generate app/artifacts/
compute_metrics.py    ← recomputes app/artifacts/metrics.json
```

| # | Model | How it runs in the app |
|---|-------|------------------------|
| 1 | TF-IDF Cosine Similarity | bundled, instant |
| 2 | TF-IDF + Logistic Regression | bundled, instant |
| 3 | Retrieval + LogReg fallback | bundled, instant |
| 4 | PyTorch MLP Scorer | bundled, instant |
| 5 | Mini Transformer (scratch) | bundled, instant |
| 6 | Sentence-BERT MiniLM-L6-v2 | downloads pretrained weights from Hugging Face on first use (zero-shot, no fine-tuning) |
| 8 | Sentence-BERT BGE-base | same as above |
| 7 | RAG + GPT-4o-mini | needs **your own** OpenAI API key, entered in the sidebar each session |

**Note on accuracy:** these models are trained on your competition's
specific question style (technical/academic MCQs). They won't be
meaningfully accurate on generic trivia unrelated to that domain — that's
expected, not a bug.

**Note on Model 7:** the notebook had an OpenAI key hardcoded in plaintext.
I removed it. That key (and the WandB key in cell 13) should be rotated —
anything pasted into a notebook is effectively public once shared. The app
now asks each user for their own key via a password-masked field; it's
used only for that session's API calls and never written to disk.

**Not included:** Model 7's local-LLM variant (Qwen2.5-1.5B) is skipped —
downloading and running a multi-GB model is impractical for most
Streamlit deployments (Streamlit Community Cloud gives ~1GB RAM). If you
need it, run that piece separately on a GPU machine.

## Run locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Artifacts are already generated and included — no training needed to run
the app. If you ever want to regenerate them (e.g. after changing data):

```bash
# from the project root, one level above app/
python train_artifacts.py     # retrains Models 1-5, ~5-10 min on CPU
python compute_metrics.py     # rebuilds the comparison table
```

## Deploy to Streamlit Community Cloud (free)

1. Push this whole `app/` folder (plus `requirements.txt` inside it) to a
   **public or private GitHub repo**.
   - `artifacts/` (~35MB) is small enough to commit directly to Git — no
     Git LFS needed.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**.
3. Point it at your repo, branch, and set the main file path to `app.py`.
4. Deploy. First load will be slower while dependencies install; after
   that it's cached.
5. If you'll use Model 7, don't hardcode a key anywhere — let users type
   their own into the sidebar field, as the app already does.

### Resource notes for Community Cloud (1GB RAM, no GPU)
- Models 1–5: fine, lightweight.
- Models 6/8 (sentence-transformers + torch): adds real memory pressure.
  If the app crashes on first use of these, consider removing them from
  `MODEL_CATALOG` in `app.py`, or deploying on a host with more RAM
  (Render, Railway, a small VM, HF Spaces with more memory).
- Model 7: just an API call, cheap on memory — safe to keep.

## Alternative hosts
- **Hugging Face Spaces** (Streamlit SDK) — generous free tier, good fit
  if you keep Models 6/8.
- **Render / Railway** — simple Docker/native Python deploys if you want
  more control over RAM/CPU.

## App features
- **🔮 Try a question** — type a prompt + 5 options, get a ranked
  prediction with a confidence bar chart.
- **📄 Batch predict (CSV)** — upload a CSV (`prompt, A, B, C, D, E`, optional
  `answer`), get predictions for all rows plus a downloadable
  `submission.csv`. If `answer` is present, MAP@3 / Top-1 are shown.
- **📊 Model comparison** — validation-set MAP@3 / Top-1 / Top-3 for
  Models 1–5 side by side.
