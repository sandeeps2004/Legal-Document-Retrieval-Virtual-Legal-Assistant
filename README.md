# Legal Document Retrieval & Virtual Legal Assistant

AI-powered legal assistant for Indian law using RAG (Retrieval-Augmented Generation). Search across IPC, BNS 2023, Constitution, CrPC, 24,000+ legal Q&A pairs, and 170,000+ court judgments.

## Architecture

```
┌─────────────────────────────────┐
│   Next.js Frontend (3D UI)      │  ← localhost:3000
│   Three.js + Tailwind + SSE     │
└──────────────┬──────────────────┘
               │ HTTP / SSE
               ▼
┌─────────────────────────────────┐
│   FastAPI Backend                │  ← localhost:8000
│                                  │
│   ┌───────────────────────────┐ │
│   │  Retriever (pgvector+BM25)│ │
│   │  Cross-encoder reranker   │ │
│   │  Gemini 2.5 Flash (LLM)  │ │
│   └───────────────────────────┘ │
└──────────────┬──────────────────┘
               │ PostgreSQL
               ▼
┌─────────────────────────────────┐
│   Neon (pgvector)               │  ← Cloud database
│   384-dim embeddings            │
│   200K+ legal document chunks   │
└─────────────────────────────────┘
```

## Tech Stack

| Layer      | Technology                                    |
|------------|-----------------------------------------------|
| Frontend   | Next.js 15, Three.js, React Three Fiber, Tailwind, Framer Motion |
| Backend    | FastAPI, Python 3.12                          |
| LLM        | Google Gemini 2.5 Flash                       |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2)      |
| Reranker   | cross-encoder/ms-marco-MiniLM-L-6-v2         |
| Vector DB  | Neon PostgreSQL + pgvector                    |
| Search     | Hybrid (semantic + BM25) with cross-encoder reranking |

## Features

- **Legal Search**: Ask any question about Indian law with cited sources
- **Personal Legal Assistant**: Describe a problem, get rights + applicable laws + recommended actions
- **Hybrid Search**: Combines semantic vector search with BM25 keyword search
- **Cross-encoder Reranking**: Reranks results for precision
- **Streaming Responses**: Real-time token-by-token answer generation via SSE
- **3D Interactive UI**: Animated star field and floating orbs background
- **Source Citations**: Every answer includes source documents with relevance scores

## Data Sources

- Indian Penal Code (IPC)
- Bharatiya Nyaya Sanhita 2023 (BNS)
- Constitution of India
- Code of Criminal Procedure (CrPC)
- Transfer of Property Act, RTI Act, Consumer Protection Act, and more
- 24,000+ legal Q&A pairs
- 170,000+ Supreme Court and High Court judgments

## Setup

### 1. Backend

```bash
cd legal-rag
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and DATABASE_URL
```

### 2. Database (Neon)

Create a free project at [neon.tech](https://neon.tech), then run:

```bash
python3 -c "
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
cur.execute('''CREATE TABLE IF NOT EXISTS legal_chunks (
  id bigserial PRIMARY KEY, text text NOT NULL,
  embedding vector(384), source text NOT NULL,
  collection text NOT NULL, chunk_hash text UNIQUE NOT NULL,
  metadata jsonb DEFAULT \\'{}\\', created_at timestamptz DEFAULT now())''')
cur.execute('CREATE INDEX IF NOT EXISTS legal_chunks_collection_idx ON legal_chunks (collection)')
conn.commit()
print('Done')
conn.close()
"
```

### 3. Ingest Data

```bash
python3 ingest/embedder.py
```

### 4. Run

```bash
# Terminal 1: Backend
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd ../frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Project Structure

```
legal-rag/
├── api/main.py              # FastAPI backend (REST + SSE streaming)
├── ingest/
│   ├── csv_loader.py         # Load + chunk CSV Q&A dataset
│   ├── pdf_loader.py         # Extract + chunk PDFs (PyMuPDF)
│   ├── jsonl_loader.py       # Load + chunk JSONL datasets
│   └── embedder.py           # Embed chunks + store in pgvector
├── rag/
│   ├── retriever.py          # Hybrid search (pgvector + BM25) + reranker
│   ├── generator.py          # Gemini answer generation with citations
│   └── pipeline.py           # Full RAG pipeline
├── assistant/
│   └── legal_assistant.py    # Personal legal assistant mode
├── config.py
├── requirements.txt
└── .env

frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx          # Main UI (search + assistant modes)
│   │   └── globals.css
│   ├── components/
│   │   ├── Scene3D.tsx       # Three.js 3D background
│   │   ├── ConfidenceBadge.tsx
│   │   ├── SourceChips.tsx
│   │   └── TypingIndicator.tsx
│   └── lib/
│       └── api.ts            # API client + SSE streaming
└── .env.local
```

## Disclaimer

This is AI-generated legal information, not professional legal advice. Always consult a qualified lawyer for your specific situation.
