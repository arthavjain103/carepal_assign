# Chunking strategy

Data: `data/cleaned_pages.json` — 31 scraped pages from thecarepal.com (113k characters).

## What the data looks like

| Observation | Consequence for chunking |
|---|---|
| Pages are Markdown with real headings (`#`…`####`); most facts live under one heading | Split on headings, not on a character count |
| Very uneven pages: 160 chars (login page) … 36,776 chars (privacy policy); median ≈ 1.5k | One fixed size is wrong for most pages; whole pages are far too big for the privacy/terms pages |
| 14 blog posts of ≈1.4–1.5k chars, one idea each | Should stay as one chunk |
| Many short H3/H4 blocks (3–5 bullet items) under one parent | Must be merged, otherwise 100–200 char fragments |
| Noise: WhatsApp icon link on all 31 pages, social-icon headings, carousel blocks scraped twice, `0` counters, empty `##` headings, video-tag text | Remove before chunking, or noise gets indexed and retrieved |
| A heading often *is* the question ("What Elderly Care includes?", "Retention of Your Personal Data") | Keep the heading with its text, and add it as context |

## Strategies compared

Benchmark: 33 questions written from the data (gold page + gold answer phrase), run through BM25, TF-IDF cosine and RRF hybrid of both. A hit = the chunk is from the right page **and** contains the answer phrase.
Numbers below are the **hybrid** retriever (hit@3 / MRR).

| Strategy | Chunks | Avg chars | Hit@3 | MRR | Verdict |
|---|---|---|---|---|---|
| A. Fixed 700/150 (old setting) | 247 | 495 | 0.79 | 0.70 | cuts mid-section, 23 fragments <150 chars, noise kept |
| B. Fixed 1200/150 | 145 | 805 | 0.79 | 0.74 | still blind to headings |
| C. Whole page | 31 | 3,662 | 0.94 | 0.87 | see below: not usable |
| D. Semantic (similarity breakpoints) | 291 | 388 | 0.73 | 0.59 | unstable on short, list-like web text |
| E. Header split only | 283 | 400 | 0.73 | 0.63 | 119 tiny chunks, 4.7k-char outliers |
| **F. Structure-aware (chosen)** | **187** | **705** | **0.85** | **0.79** | |

Ablations of F: without the title/section prefix MRR drops 0.79 → 0.70; cleaning alone on fixed-1200 gives 0.77; max size 1000–1600 is a plateau (0.75–0.79), 800 and 2000 are worse.

**Why not whole page (C)?** It "wins" the benchmark only because a page-level hit is easy for keyword search. In the real pipeline the embedding model (`multilingual-e5-base`) and the cross-encoder reranker both truncate at 512 tokens (≈2,000 chars), so 57% of the corpus text would be invisible to them, and the top-3 context sent to the LLM would average ≈18k characters (vs ≈2k).

**Why not semantic chunking (D)?** It needs an embedding call per sentence at ingest time, and on short marketing/FAQ-style text its breakpoints were unstable; it also ignores the heading structure that the pages already give us for free. (Caveat: I could not reach the HuggingFace embedding model from my sandbox, so D used TF-IDF similarity as a stand-in; a real-embedding version might do somewhat better, but still not beat explicit headings.)

## Chosen: structure-aware chunking (`ingestion/chunker.py`)

1. **Clean** the scraped text (icon links, social headings, empty headings, `0` counters, duplicated blocks).
2. **Split on Markdown headings** and keep the heading breadcrumb.
3. **Pack** neighbouring small sections (< 300 chars) together up to a 1200-char cap.
4. **Split only oversize sections** (privacy policy, terms, home page) with a recursive splitter (paragraph → bullet → line → sentence → word), 150-char overlap, and repeat the section heading in each part.
5. **Prefix every chunk** with `Page: <title>` and `Section: A > B > C`, and store `url`, `title`, `category`, `section`, `chunk_index`, `chunk_id` as metadata.
6. Skip pages < 200 chars after cleaning (the "Sign in" page).

Why 1200 chars: ≈300 tokens, comfortably inside the 512-token limit of the embedder and reranker, big enough for a full section or a whole blog post, small enough that 5 chunks (≈3.5k chars) is a cheap LLM context.

Result: 187 chunks, 160–1,358 chars, 0 noise chunks, 0 duplicates, 0 fragments < 150 chars.

## Limitations

- The 33-question set was written by me from the same data; it is a sanity benchmark, not a held-out test. Retrieval here is lexical (BM25 + TF-IDF) because real embeddings were not reachable, so absolute numbers will differ with Pinecone + e5. The ranking of strategies is driven mostly by structure (fragments, noise, context), which does not depend on the retriever.
- The e5 models are trained with `query: ` / `passage: ` prefixes; the current `embedder.py` does not add them. Adding them is a separate, usually positive, change.
