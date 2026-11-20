"""
Structure-aware chunking for the scraped CarePal pages.

Strategy (chosen after benchmarking, see CHUNKING.md):
  1. Clean scraping noise (icon links, duplicated carousel blocks, empty headings...).
  2. Split every page on its Markdown headings, keeping the heading breadcrumb
     (e.g. "Elderly Care > What Elderly Care includes? > Daily Living Support").
  3. Pack neighbouring tiny sections together (never above MAX_CHUNK_CHARS) so
     chunks are not 1-line fragments.
  4. Only sections that are still too big are split further with a recursive
     splitter (paragraph -> bullet -> line -> sentence -> word), with overlap.
  5. Every chunk is prefixed with "Page title / Section" so it is
     self-explanatory to BM25, the embedding model, the reranker and the LLM.
"""
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, CHUNK_OVERLAP, MIN_PAGE_CHARS

# --------------------------------------------------------------------------- #
# 1. Cleaning
# --------------------------------------------------------------------------- #
_NOISE_LINE = re.compile(
    r"^\s*("
    r"\[\]\(.*\)"                                   # icon-only links, e.g. the WhatsApp button
    r"|Your browser does not support the video tag\.?"
    r"|←\s*Back to all articles"
    r"|0"                                           # animated counters scraped as "0"
    r"|#{1,6}\s*"                                   # empty headings
    r"|#{1,6}\s*(Twitter|Facebook|App\s?store|Play\s?store|Linkedin|Youtube|Instagram)\s*"
    r")\s*$",
    re.IGNORECASE,
)
# Headings nested in bullets ("* #### Foo") -> real headings
_BULLET_HEADING = re.compile(r"^\s*[*\-]\s*(#{1,6})\s*")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


_EMPTY_LINK = re.compile(r"\[\]\([^)]*\)")


def clean_text(text: str) -> str:
    text = _EMPTY_LINK.sub("", text)          # inline icon-only links
    lines = []
    for line in text.replace("\r", "").split("\n"):
        line = _BULLET_HEADING.sub(r"\1 ", line)
        if _NOISE_LINE.match(line):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)

    # Drop paragraph blocks that repeat within the page (carousels are scraped twice).
    seen, blocks = set(), []
    for block in re.split(r"\n\s*\n", text):
        key = re.sub(r"\s+", " ", block).strip().lower()
        if not key:
            continue
        if len(key) >= 30 and key in seen:
            continue
        seen.add(key)
        blocks.append(block.strip())
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 2. Split by headings
# --------------------------------------------------------------------------- #
def split_sections(text: str):
    """Return [(breadcrumb_list, section_text_including_its_heading_line)]."""
    sections, stack, buf = [], [], []   # stack: [(level, heading)]
    path_of_buf = []

    def flush():
        body = "\n".join(buf).strip()
        # a section that is only a heading line carries no information by itself;
        # its heading is still kept in the breadcrumb of the sections below it
        if body and not (len(buf) == 1 and _HEADING.match(buf[0])):
            sections.append((list(path_of_buf), body))

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            buf.clear()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path_of_buf = [t for _, t in stack]
        buf.append(line)
    flush()
    return sections


# --------------------------------------------------------------------------- #
# 3 + 4. Pack small sections, split big ones
# --------------------------------------------------------------------------- #
def _split_big_section(body: str):
    """Split an oversize section; every part keeps the section's heading line."""
    first, _, rest = body.partition("\n")
    heading = first if _HEADING.match(first) else ""
    if not heading:
        rest = body
    budget = MAX_CHUNK_CHARS - (len(heading) + 1 if heading else 0)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=budget,
        chunk_overlap=min(CHUNK_OVERLAP, budget // 4),
        separators=["\n\n", "\n* ", "\n", ". ", " ", ""],
    )
    parts = splitter.split_text(rest)
    return [f"{heading}\n{p}" if heading else p for p in parts]


def _short_title(title: str) -> str:
    return (title or "").split("|")[0].strip()


def chunk_page(doc: Document):
    text = clean_text(doc.page_content)
    if len(text) < MIN_PAGE_CHARS:          # e.g. the "Sign in" page
        return []

    page_title = _short_title(doc.metadata.get("title", ""))
    pieces = []                              # (section_path, text)

    for path, body in split_sections(text):
        if len(body) <= MAX_CHUNK_CHARS:
            pieces.append((path, body))
        else:
            for part in _split_big_section(body):
                pieces.append((path, part))

    # pack: absorb a following piece while the current one is still small
    packed = []
    for path, body in pieces:
        if packed and len(packed[-1][1]) < MIN_CHUNK_CHARS \
                and len(packed[-1][1]) + 2 + len(body) <= MAX_CHUNK_CHARS:
            packed[-1] = (packed[-1][0], packed[-1][1] + "\n\n" + body)
        else:
            packed.append((path, body))
    # a small tail goes back to the previous chunk if it fits
    if len(packed) > 1 and len(packed[-1][1]) < MIN_CHUNK_CHARS \
            and len(packed[-2][1]) + 2 + len(packed[-1][1]) <= MAX_CHUNK_CHARS:
        tail = packed.pop()
        packed[-1] = (packed[-1][0], packed[-1][1] + "\n\n" + tail[1])

    chunks = []
    for i, (path, body) in enumerate(packed):
        section = " > ".join(path)
        header = f"Page: {page_title}" + (f"\nSection: {section}" if section else "")
        meta = {
            **doc.metadata,
            "section": section,
            "chunk_index": i,
            "chunk_id": f"{doc.metadata.get('content_hash', doc.metadata.get('url', ''))[:12]}-{i}",
        }
        chunks.append(Document(page_content=f"{header}\n\n{body}", metadata=meta))
    return chunks


def chunk_documents(documents):
    chunks = []
    for doc in documents:
        chunks.extend(chunk_page(doc))
    sizes = [len(c.page_content) for c in chunks]
    if sizes:
        print(f"Chunks: {len(chunks)} | chars min/avg/max: "
              f"{min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")
    return chunks
