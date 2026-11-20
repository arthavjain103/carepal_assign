import json
from langchain_core.documents import Document


def load_documents(path="data/cleaned_pages.json"):
    """Load the scraped pages (list of {url, title, category, text, ...}) as Documents."""
    with open(path, encoding="utf-8") as f:
        pages = json.load(f)

    docs = [
        Document(
            page_content=p["text"],
            metadata={
                "url": p.get("url", ""),
                "title": p.get("title", ""),
                "category": p.get("category", ""),
                "content_hash": p.get("content_hash", ""),
            },
        )
        for p in pages
        if p.get("text", "").strip()
    ]
    print(f"Loaded {len(docs)} pages")
    return docs
