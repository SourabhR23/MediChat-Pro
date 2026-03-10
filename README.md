# 🏥 MediChat Pro

**An AI-powered Medical Document Assistant** that processes patient history PDFs and enables intelligent, RAG-based conversational querying — built with a production-ready FastAPI + Streamlit architecture.

## 🌐 Live Demo

Check the deployed application here:

[Open MediChat Pro](https://medichat-pro-u3u4.onrender.com/)

---

## 📌 Project Overview

MediChat Pro allows users to upload patient medical history PDFs and chat with the data in natural language. The system retrieves relevant chunks from an embedded vector store and generates structured, safety-aware responses using an LLM.

Built as a **functional RAG system**.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│              User (Browser)              │
└──────────────────┬───────────────────────┘
                   │
       ┌───────────▼───────────┐
       │  Streamlit Frontend   │  main.py  :8501
       └───────────┬───────────┘
                   │ HTTP REST
       ┌───────────▼───────────┐
       │   FastAPI Backend     │  api.py    :8000
       └──┬──────────┬─────────┘
          │          │          │
   ┌──────▼──┐  ┌────▼──┐  ┌───▼──────┐
   │ChromaDB │  │  S3   │  │          │
   │(Vectors)│  │(PDFs) │  │  (LLM)   │
   └─────────┘  └───────┘  └──────────┘
```

**Key design decision**: FastAPI and Streamlit run as separate services for independent deployment, testability, and scalability.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- ChromaDB Cloud account
- Euri AI API key / OpenAI API key
- HuggingFace API token
- AWS S3 bucket (optional)

### Installation

```bash
git clone <repo-url>
cd medichat-pro
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
EURI_API_KEY=your_euri_api_key
LLM_MODEL=gpt-4.1-nano

HF_API_TOKEN=your_huggingface_token
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=medibot
CHROMA_COLLECTION=medical_documents

S3_ACCESS_KEY=your_aws_access_key
S3_SECRET_KEY=your_aws_secret_key
S3_BUCKET_NAME=medibot-euron-2025
S3_REGION=ap-south-1

EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECEIVER=default_receiver@example.com
```

### Running

```bash
# Terminal 1 — Backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
streamlit run main.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`

---

## 📂 Project Structure

```
medichat-pro/
├── api.py                    # FastAPI backend — 12 REST endpoints
├── main.py                  # Streamlit frontend — chat UI
├── app/
│   ├── config.py             # Env variable loading
│   ├── pdf_utils.py          # PDF extraction + cleaning
│   ├── vectorstore_utils.py  # ChromaDB + HuggingFace embeddings
│   ├── chat_utils.py         # Euri AI LLM client
│   ├── email_utils.py        # SMTP email reports
│   └── s3_utils.py           # AWS S3 management
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/status` | Vectorstore state |
| POST | `/init` | Load from ChromaDB |
| POST | `/documents/upload` | Upload PDFs → embed |
| POST | `/documents/process-s3` | Re-embed from S3 |
| GET | `/documents/s3-list` | List S3 PDFs |
| DELETE | `/documents/clear` | Wipe ChromaDB |
| POST | `/chat` | RAG chat + insights |
| POST | `/email/send-report` | Send analytics email |
| POST | `/email/support-ticket` | Create support ticket |
| POST | `/email/save-session` | Save session email |
| POST | `/email/test` | SMTP test |

---

## 💬 Chat Commands

| Type in chat | Action |
|---|---|
| `send report to user@email.com` | Email full analysis |
| `support ticket` | Create support ticket |
| `process s3` | Embed all S3 docs |
| `save session` | Email session summary |

---

## 📊 RAG Insight Engine

Every response includes a **RAG Quality Panel** with 9 patient-safety metrics:

| Metric | What it measures |
|--------|-----------------|
| Confidence Score | Weighted composite of all below (0–100%) |
| Patient Match Score | % chunks containing the queried patient's name |
| Field Coverage | Standard medical fields found in retrieved context |
| Entity Faithfulness | Medical entities from context echoed in response |
| Critical Field Safety | Allergy/medication/diagnosis always surfaced |
| Contamination Risk | Multiple patients' data mixing in chunks |
| Age/Name Consistency | Age in chunks vs. age stated in response |
| Intent Coverage | Response addresses what was actually asked |
| Avg Chunk Relevance | Token-overlap between query and each chunk |

---

## 🛡️ Patient Safety Design

1. **Allergy First** — Allergy info always surfaced; if missing, explicitly flagged
2. **Patient Isolation** — Contamination metric detects cross-patient data leakage
3. **No Hallucination** — LLM prompted to never invent medication details
4. **Missing Data Protocol** — Absent fields stated explicitly, never implied normal

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI  |
| Vector Store | ChromaDB Cloud 1.0 |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | `gpt-4.1-nano` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| PDF Parsing | pypdf 3.17 |
| Storage | AWS S3 |
| Email | Python SMTP (Gmail) |

---

## 🚢 Deployment (Render)

```bash
# Backend start command
uvicorn api:app --host 0.0.0.0 --port $PORT

# Frontend start command
streamlit run main2.py --server.address 0.0.0.0 --server.port $PORT
```

Set `API_BASE_URL` environment variable on the Streamlit service pointing to your deployed FastAPI URL.

---

## 🔮 Roadmap

- [ ] RAGAS evaluation framework
- [ ] CrossEncoder re-ranking
- [ ] pytest test coverage
- [ ] Section-aware chunking
- [ ] Per-user document isolation (JWT + metadata filters)

---

> ⚠️ For informational and educational purposes only. Not a substitute for professional medical advice.