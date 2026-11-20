import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Chunking (see CHUNKING.md) ─────────────────────────────
MAX_CHUNK_CHARS = 1200   # hard cap, ~300 tokens: fits e5-base (512) and the MiniLM reranker (512)
MIN_CHUNK_CHARS = 300    # smaller sections get packed with their neighbour
CHUNK_OVERLAP = 150      # only used when one section must be split mid-way
MIN_PAGE_CHARS = 200     # pages shorter than this after cleaning are skipped (login page etc.)

TOP_K = 5
