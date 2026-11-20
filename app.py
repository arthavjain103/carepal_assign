import re
import pickle
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pro.semantic import SemanticSearch
from pro.hybrid import CombinedSearch
from llm.chat import generate_answer_ex  # generate_answer_ex(question, context) -> (answer, signal)
import db


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


hybrid_search: CombinedSearch | None = None

URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def extract_sources(results, max_sources: int = 3) -> list[str]:
    """Unique page URLs found in the retrieved chunks, in rank order."""
    sources: list[str] = []
    for _, doc in results:
        m = URL_RE.search(doc)
        if m:
            url = m.group(0).rstrip(".,")
            if url not in sources:
                sources.append(url)
        if len(sources) >= max_sources:
            break
    return sources


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at startup; release on shutdown."""
    global hybrid_search

    logger.info("Initializing SQLite escalation DB …")
    db.init_db()
    logger.info("Escalation DB ready.")

    logger.info("Loading BM25 model from data/bm25.pkl …")
    with open("data/bm25.pkl", "rb") as f:
        keyword_search = pickle.load(f)
    logger.info("BM25 loaded.")

    logger.info("Connecting to Pinecone (SemanticSearch) …")
    semantic_search = SemanticSearch()
    logger.info("Pinecone connected.")

    logger.info("Building CombinedSearch (Hybrid RAG) pipeline …")
    hybrid_search = CombinedSearch(
        keyword_search=keyword_search,
        vector_search=semantic_search,
    )
    logger.info("Pipeline ready — server is live.")

    yield  # application runs here

    logger.info("Shutting down — cleaning up resources.")
    hybrid_search = None


app = FastAPI(
    title="CarePal Hybrid RAG Chatbot",
    description="POST /search  →  hybrid BM25 + Pinecone retrieval + Groq answer",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    """Request body: { "query": "..." }"""
    query: str = ""

    def get_query(self) -> str:
        return self.query.strip()


class SearchResponse(BaseModel):
    answer: str
    chunks: list[dict] = []     # [{"text": "...", "score": 0.85}, ...]
    sources: list[str] = []
    signal: str | None = None   # 'NOT_FOUND' | 'ESCALATE' | None


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "CarePal RAG API is running. See /docs"}


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """Simple liveness probe."""
    return HealthResponse(status="ok")


@app.post("/search", response_model=SearchResponse, tags=["RAG"])
async def search(request: SearchRequest):

    try:
        query = request.get_query()
        if not query:
            return SearchResponse(answer="Please provide a valid question.")

        logger.info("Received query: %s", query)

        if hybrid_search is None:
            return SearchResponse(answer="Search service is not ready.")

        # Step 1-3: Hybrid retrieval (BM25 + Pinecone + RRF + Cross-encoder)
        # hybrid_search.search returns: list of (score, doc_text)
        results = hybrid_search.search(query=query, top_n=5)

        # Chunks for the UI (sidebar "Retrieved Chunks")
        chunks = [
            {"text": doc, "score": float(score)}
            for score, doc in results
        ]

        # Step 4: Build context from retrieved chunks
        context = "\n\n---\n\n".join(doc for score, doc in results)

        if not context.strip():
            logger.warning("No context retrieved for query: %s", query)
            db.log_event(question=query, answer="I couldn't find this information.", signal="NOT_FOUND")
            return SearchResponse(answer="I couldn't find this information.", signal="NOT_FOUND")

        # Step 5: Generate final answer via Groq
        answer, signal = generate_answer_ex(question=query, context=context)

        # Log escalation / not-found events for admin review
        if signal in ("NOT_FOUND", "ESCALATE"):
            db.log_event(question=query, answer=answer, signal=signal)

        # Cite only when the model actually answered from the content
        not_found = signal == "NOT_FOUND"
        sources = [] if not_found else extract_sources(results)

        logger.info("Answer generated successfully. Signal: %s, Sources: %s", signal, sources)
        return SearchResponse(answer=answer, chunks=chunks, sources=sources, signal=signal)

    except Exception as exc:
        logger.error("Error processing query '%s': %s", request.query, exc, exc_info=True)
        return SearchResponse(answer="Sorry, I couldn't retrieve the information right now.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)