# AI and Data Science Portfolio

Three small projects that show an end-to-end workflow: retrieval-augmented generation (RAG), supervised learning, and unsupervised learning. All data is public or synthetic.

| Project | What it shows | Tools |
|---|---|---|
| [01 RAG metric assistant](01_rag_metric_assistant) | Answers metric questions from approved definitions only, served as a FastAPI service. Retrieval finds the right definition in the top 3 for 15 of 15 test questions, and I confirmed the model choice on held-out questions | Python, sentence-transformers, FAISS, LLM API, FastAPI, pytest |
| [02 Credit default model](02_credit_default_model) | Predicts credit card default (22% base rate). Gradient boosting reached 0.78 AUC vs 0.73 for a logistic baseline, with threshold tuning for a realistic outreach goal | pandas, scikit-learn |
| [03 Customer segmentation](03_customer_segmentation) | K-means segments on 30,000 customers. Four stable segments with default rates from 11% to 36%, though the model never saw the label. Isolation Forest flags unusual customers | pandas, scikit-learn |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Each project folder has its own README with run instructions and results.

## Notes
- API keys live in a local `.env` file that is never committed. See `.env.example`.
- The credit card dataset is not stored in this repo. Download it from [UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients); the project README explains the one-step cleanup.