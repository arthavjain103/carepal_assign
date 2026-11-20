# CarePal RAG Chatbot

Hybrid retrieval-augmented chatbot over the scraped thecarepal.com pages: BM25 + Pinecone semantic search, reranking, and Groq answer generation.

## Project Structure

```text
├── app.py                 # FastAPI server (/search, /health)
├── main.py                # CLI chatbot
├── ingest.py              # Chunk, upload to Pinecone, build BM25 index
├── CHUNKING.md            # Why the chunking strategy was chosen (benchmark)
├── scraper.py             # Old LendingKart scraper (not used for the CarePal data)
├── reranker.py            # Cross-encoder reranker
├── config.py              # Config and env loader
├── data/cleaned_pages.json # Scraped CarePal pages (input)
├── data/bm25.pkl          # Created by ingest.py
├── llm/chat.py            # Groq prompt and answer generation
├── pro/                   # keyword_based.py, semantic.py, hybrid.py
└── ingestion/             # json_loader, chunker (structure-aware), embedder, pinecone_upload
```

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` (see `.env.example`):

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=your-pinecone-index-name
GROQ_API_KEY=your-groq-api-key
```

## Build the knowledge base (required before first run)

```bash
python ingest.py --reset   # --reset clears the Pinecone index first (old data / old chunks)
```

This chunks `data/cleaned_pages.json`, uploads vectors to Pinecone and writes `data/bm25.pkl`.

## Run

CLI chatbot:

```bash
python main.py
```

API server:

```bash
python app.py        # http://localhost:8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "What are the eligibility criteria for a business loan?"}'
```

## How it works

1. BM25 and Pinecone semantic search each retrieve candidate chunks.
2. Results are fused (RRF) and reranked with a cross-encoder.
3. The top chunks are passed as context to Groq (`openai/gpt-oss-120b`), which answers only from that context.

## Tech stack

Groq LLM, Pinecone, `rank-bm25`, HuggingFace `sentence-transformers`, FastAPI, Python.
