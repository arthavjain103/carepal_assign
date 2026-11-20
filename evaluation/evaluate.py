"""
Ragas evaluation script for the CarePal RAG chatbot (Ragas 0.4.x).

Runs every question in evaluation/test_questions.json through the hybrid RAG
pipeline (BM25 + Pinecone + RRF + cross-encoder rerank + Groq), then scores
the results with Ragas metrics:

  * Faithfulness
  * Answer Relevancy
  * Context Precision
  * Context Recall
"""

import asyncio
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# Add project root to sys.path so local packages (pro, llm, ingestion) import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pro.semantic import SemanticSearch
from pro.hybrid import CombinedSearch
from llm.chat import generate_answer

TEST_QUESTIONS_PATH = ROOT / "evaluation" / "test_questions.json"
BM25_PATH = ROOT / "data" / "bm25.pkl"
RESULTS_PATH = ROOT / "evaluation" / "ragas_results.json"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def initialize_pipeline():
    """Load BM25, connect to Pinecone, and build the hybrid search pipeline."""
    print("Loading BM25 index …")
    if not BM25_PATH.exists():
        print(f"ERROR: {BM25_PATH} not found. Run `python ingest.py --reset` first.")
        sys.exit(1)

    with open(BM25_PATH, "rb") as f:
        keyword_search = pickle.load(f)
    print("BM25 loaded.")

    print("Connecting to Pinecone …")
    semantic_search = SemanticSearch()
    print("Pinecone connected.")

    print("Initializing Hybrid Search (BM25 + Pinecone + RRF + reranker) …")
    hybrid_search = CombinedSearch(
        keyword_search=keyword_search,
        vector_search=semantic_search,
    )
    print("Pipeline ready.\n")
    return hybrid_search


def run_rag(hybrid_search, query, top_n=3, max_retries=3):
    """
    Run one query through the RAG pipeline.

    Returns:
        answer (str)          – LLM-generated answer
        retrieved_contexts (list[str]) – top-N chunk texts
    """
    results = hybrid_search.search(query=query, top_n=top_n)

    retrieved_contexts = [doc for score, doc in results]

    context = "\n\n---\n\n".join(retrieved_contexts)

    # Retry with exponential backoff for Groq rate limits (HTTP 429)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            answer = generate_answer(query, context)
            return answer, retrieved_contexts
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "rate_limit_exceeded" in msg or "429" in msg:
                wait = min(2 ** attempt, 30)
                print(f"  [rate limited, retry {attempt}/{max_retries} in {wait}s]")
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"generate_answer failed after {max_retries} retries: {last_error}")


async def score_with_retry(name, coro_factory, max_retries=4):
    """Score one metric; retry on rate limits, return NaN on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            result = await coro_factory()
            value = getattr(result, "value", result)
            return float(value) if value is not None else float("nan")
        except Exception as exc:
            msg = str(exc)
            if "rate_limit" in msg or "429" in msg:
                wait = min(5 * 2 ** (attempt - 1), 60)
                print(f"    [{name}] rate limited, retry {attempt}/{max_retries} in {wait}s")
                await asyncio.sleep(wait)
                continue
            print(f"    [{name}] failed: {msg[:200]}")
            return float("nan")
    print(f"    [{name}] gave up after {max_retries} retries")
    return float("nan")


async def score_row(metrics, row):
    """Score one dataset row with all four metrics (sequentially, to spare Groq TPM)."""
    q = row["user_input"]
    a = row["response"]
    ctx = row["retrieved_contexts"]
    ref = row["reference"]

    scores = {}

    scores["faithfulness"] = await score_with_retry(
        "faithfulness",
        lambda: metrics["faithfulness"].ascore(
            user_input=q, response=a, retrieved_contexts=ctx
        ),
    )
    scores["answer_relevancy"] = await score_with_retry(
        "answer_relevancy",
        lambda: metrics["answer_relevancy"].ascore(user_input=q, response=a),
    )

    if ref:
        scores["context_precision"] = await score_with_retry(
            "context_precision",
            lambda: metrics["context_precision"].ascore(
                user_input=q, reference=ref, retrieved_contexts=ctx
            ),
        )
        scores["context_recall"] = await score_with_retry(
            "context_recall",
            lambda: metrics["context_recall"].ascore(
                user_input=q, retrieved_contexts=ctx, reference=ref
            ),
        )
    else:
        # These two metrics need a reference answer.
        scores["context_precision"] = float("nan")
        scores["context_recall"] = float("nan")

    return scores


def mean_ignore_nan(values):
    clean = [v for v in values if not (v is None or math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def nan_to_none(x):
    return None if (isinstance(x, float) and math.isnan(x)) else x


async def evaluate_rows(rows):
    # ── Ragas evaluator LLM (Groq via OpenAI-compatible ASYNC client) ─────
    print("Setting up Ragas evaluator LLM (openai/gpt-oss-20b via Groq) …")
    groq_client = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory("openai/gpt-oss-20b", client=groq_client)

    # ── Ragas evaluator embeddings (multilingual-e5-base) ─────────────────
    print("Setting up Ragas evaluator embeddings (intfloat/multilingual-e5-base) …")
    evaluator_embeddings = HuggingFaceEmbeddings(
        model="intfloat/multilingual-e5-base"
    )

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }

    per_question = []
    for i, row in enumerate(rows, start=1):
        print(f"\nScoring [{i}/{len(rows)}]: {row['user_input']}")
        scores = await score_row(metrics, row)
        print("  " + " | ".join(f"{k}={v:.3f}" if not math.isnan(v) else f"{k}=NaN"
                                for k, v in scores.items()))
        per_question.append(scores)
        await asyncio.sleep(1.0)

    return per_question


def main():
    # ── Load test questions ───────────────────────────────────────────────
    print(f"Loading test questions from {TEST_QUESTIONS_PATH} …")
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    print(f"Loaded {len(test_questions)} questions.\n")

    # ── Initialize RAG pipeline ───────────────────────────────────────────
    hybrid_search = initialize_pipeline()

    rows = []
    for i, item in enumerate(test_questions, start=1):
        question = item["question"]
        reference = item.get("reference", "")

        print("\n" + "=" * 80)
        print(f"[{i}/{len(test_questions)}] QUESTION: {question}")
        print("=" * 80)

        answer, contexts = run_rag(hybrid_search, question, top_n=3)

        # Small delay between questions to avoid Groq TPM rate limits
        if i < len(test_questions):
            time.sleep(1.5)

        print("\nANSWER:")
        print(answer)

        print("\nRETRIEVED CONTEXTS:")
        for j, ctx in enumerate(contexts, 1):
            print(f"\n--- Context {j} ---")
            print(ctx[:500] + ("…" if len(ctx) > 500 else ""))

        rows.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
        })

    print("\n" + "=" * 80)
    print("Running Ragas evaluation …")
    print("=" * 80)

    per_question = asyncio.run(evaluate_rows(rows))

    averages = {
        name: mean_ignore_nan([s[name] for s in per_question])
        for name in METRIC_NAMES
    }

    print("\n" + "=" * 80)
    print("RAGAS EVALUATION RESULTS (mean over questions, NaN ignored)")
    print("=" * 80)
    for name, value in averages.items():
        print(f"{name:20s}: {value:.4f}" if not math.isnan(value) else f"{name:20s}: NaN")

    results_path = RESULTS_PATH
    results_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "ragas_results": {k: nan_to_none(v) for k, v in averages.items()},
        "per_question": [
            {
                "user_input": row["user_input"],
                "response": row["response"],
                "reference": row["reference"],
                "scores": {k: nan_to_none(v) for k, v in scores.items()},
            }
            for row, scores in zip(rows, per_question)
        ],
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()