"""
Build the knowledge base from data/cleaned_pages.json:
  pages -> structure-aware chunks -> Pinecone vectors + BM25 index (data/bm25.pkl)

Usage:
  python ingest.py            # add chunks to the Pinecone index
  python ingest.py --reset    # wipe the Pinecone index first (use when the index
                              # still holds chunks from older data / an old chunker)
"""
import os
import pickle
import sys

from ingestion.json_loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.pinecone_upload import upload
from pro.keyword_based import KeywordSearchBM25

reset = "--reset" in sys.argv

docs = load_documents("data/cleaned_pages.json")
chunks = chunk_documents(docs)

# Pinecone (semantic index)
upload(chunks, reset=reset)

# BM25 (keyword index)
keyword_search = KeywordSearchBM25([c.page_content for c in chunks])
os.makedirs("data", exist_ok=True)
with open("data/bm25.pkl", "wb") as f:
    pickle.dump(keyword_search, f)

print("BM25 Saved Successfully")
