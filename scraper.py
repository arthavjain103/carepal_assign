import os
import re
import sys
import json
import csv
import time
import hashlib
import difflib
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urljoin, urlparse, urldefrag
import xml.etree.ElementTree as ET

import nest_asyncio
import requests
import cloudscraper
from bs4 import BeautifulSoup

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    BrowserConfig,
    CacheMode,
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from pydantic import BaseModel, Field

from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DOMAIN = "www.thecarepal.com"
BASE_URL = f"https://{BASE_DOMAIN}"

OUTPUT_DIR = Path("thecarepal_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USE_CREWAI = True
MAX_PAGES = 40                 # Increase after first successful run
MIN_CONTENT_LENGTH = 100
NEAR_DUPLICATE_THRESHOLD = 0.92
REQUEST_TIMEOUT = 25
CRAWL_DELAY_SECONDS = 1.0
CREWAI_DELAY_SECONDS = 2.0     # pause between LLM calls (Groq rate limits)

# llama-3.3-70b-versatile is retired on Groq. Models are tried in this order;
# a model that returns "not found" is skipped automatically.
# Override from Colab with e.g.:
#   os.environ["GROQ_MODELS"] = "openai/gpt-oss-20b,qwen/qwen3-32b"
GROQ_MODELS = [
    m.strip() for m in os.getenv(
        "GROQ_MODELS",
        "openai/gpt-oss-120b,"
        "openai/gpt-oss-20b,"
        "qwen/qwen3-32b,"
        "llama-3.1-8b-instant",
    ).split(",") if m.strip()
]

# Free-tier tokens-per-minute limits (used to throttle requests).
MODEL_TPM_LIMITS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": 30000,
    "openai/gpt-oss-20b": 8000,
    "openai/gpt-oss-120b": 8000,
    "qwen/qwen3-32b": 6000,
    "llama-3.1-8b-instant": 6000,
}
DEFAULT_TPM_LIMIT = 6000
REUSE_CLEANED_PAGES = True      # skip scraping if cleaned_pages.json already exists
MAX_LLM_CHARS = 6000           # page text sent to the LLM (keeps tokens low)
LLM_MAX_ATTEMPTS = 5           # attempts per model

# Start with confirmed pages. Sitemap + internal-link discovery will add
# other relevant service/policy pages found on the live website.
SEED_URLS = [
    (f"{BASE_URL}/", "Company Information"),
    (f"{BASE_URL}/contact", "Contact Information"),
    (f"{BASE_URL}/doctor", "Doctor Consultation"),
    (f"{BASE_URL}/elderly", "Elderly Care"),
    (f"{BASE_URL}/dementia", "Dementia Care"),
    (f"{BASE_URL}/consultation", "Consultation"),
    (f"{BASE_URL}/terms", "Terms & Conditions"),
    (f"{BASE_URL}/privacy", "Privacy Policy"),
]

# Relevant words used to decide whether a discovered page belongs in this KB.
RELEVANT_PATH_KEYWORDS = {
    "care", "elder", "elderly", "dementia", "alzheimer", "autism", "baby",
    "doctor", "nurse", "consult", "therapy", "mental", "emergency",
    "service", "contact", "about", "faq", "support", "privacy", "terms",
    "refund", "booking", "book", "pricing", "payment", "livein", "live-in",
}

EXCLUDED_PATH_PARTS = {
    "/wp-admin", "/login", "/signup", "/register", "/cart", "/checkout",
    "/search", "/tag/", "/author/", "/feed", ".jpg", ".jpeg", ".png",
    ".gif", ".svg", ".webp", ".css", ".js", ".zip",
}

# CarePal-specific category normalization.
CANONICAL_CATEGORIES = {
    "homepage": "Company Information",
    "company information": "Company Information",
    "contact information": "Contact Information",
    "contact": "Contact Information",
    "doctor consultation": "Doctor Consultation",
    "consultation": "Consultation",
    "elderly care": "Elderly Care",
    "dementia care": "Dementia Care",
    "alzheimer care": "Alzheimer Care",
    "autism care": "Autism Care",
    "baby care": "Baby Care",
    "live-in care": "Live-in Care",
    "livein care": "Live-in Care",
    "mental therapy": "Mental Therapy",
    "nurse consultation": "Nurse Consultation",
    "emergency care": "Emergency Care",
    "terms": "Terms & Conditions",
    "terms and conditions": "Terms & Conditions",
    "privacy": "Privacy Policy",
    "privacy policy": "Privacy Policy",
    "refund": "Refund Policy",
    "refund policy": "Refund Policy",
}

TERMINOLOGY_MAP = {
    r"\bcare pal\b": "CarePal",
    r"\bthe care pal\b": "The CarePal",
    r"\bcaregiver(s)?\b": r"caregiver\1",
    r"\bhome care\b": "home care",
    r"\bdoctor consultation(s)?\b": r"doctor consultation\1",
    r"\bnurse consultation(s)?\b": r"nurse consultation\1",
    r"\bmental therapy\b": "mental therapy",
    r"\balzheimer'?s?\b": "Alzheimer's",
    r"\bdementia care\b": "dementia care",
}

# Be conservative: don't mask CarePal's public support details or phone numbers
# automatically if they are part of the official website. This masks obvious
# personal identifiers in scraped text. Review the audit before production use.
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "aadhaar": re.compile(r"(?<![\w/+=.\-])\d{4}\s?\d{4}\s?\d{4}(?![\w])"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


# ============================================================
# LOGGING / EVENT LOOP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "run.log", mode="w", encoding="utf-8"),
    ],
    force=True,   # override handlers already installed by Colab / other libs
)
for _noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("thecarepal-kb")
logger.setLevel(logging.INFO)

nest_asyncio.apply()


# ============================================================
# URL DISCOVERY
# ============================================================

def normalize_url(url: str) -> str:
    """Normalize URL and remove fragments/tracking query parameters."""
    absolute = urljoin(BASE_URL, url.strip())
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)

    if parsed.scheme not in {"http", "https"}:
        return ""

    host = (parsed.hostname or "").lower()
    if host not in {"thecarepal.com", "www.thecarepal.com"}:
        return ""

    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/" and not path.endswith("/"):
        path += "/"

    # Ignore query parameters for canonical page URLs.
    return f"https://{BASE_DOMAIN}{path}"


def is_relevant_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()

    if any(part in path for part in EXCLUDED_PATH_PARTS):
        return False

    if path in {"", "/"}:
        return True

    return any(keyword in path for keyword in RELEVANT_PATH_KEYWORDS)


def fetch_sitemap_urls() -> List[str]:
    """Read sitemap.xml and common sitemap index locations."""
    candidates = [
        f"{BASE_URL}/sitemap.xml",
        "https://thecarepal.com/sitemap.xml",
        f"{BASE_URL}/sitemap_index.xml",
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; CarePalKnowledgeBot/1.0)"})

    found = set()

    for sitemap_url in candidates:
        try:
            response = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200 or not response.text.strip():
                continue

            root = ET.fromstring(response.content)
            locs = [
                element.text.strip()
                for element in root.iter()
                if element.tag.lower().endswith("loc") and element.text
            ]

            # If this is a sitemap index, fetch child sitemaps.
            for loc in locs:
                if "sitemap" in loc.lower() and loc.lower().endswith(".xml"):
                    try:
                        child = session.get(loc, timeout=REQUEST_TIMEOUT)
                        if child.status_code == 200:
                            child_root = ET.fromstring(child.content)
                            for child_el in child_root.iter():
                                if child_el.tag.lower().endswith("loc") and child_el.text:
                                    normalized = normalize_url(child_el.text)
                                    if normalized and is_relevant_url(normalized):
                                        found.add(normalized)
                    except Exception as exc:
                        logger.warning("Could not read child sitemap %s: %s", loc, exc)
                else:
                    normalized = normalize_url(loc)
                    if normalized and is_relevant_url(normalized):
                        found.add(normalized)

            if found:
                logger.info("Discovered %d relevant URLs from sitemap.", len(found))
                break

        except Exception as exc:
            logger.warning("Sitemap unavailable at %s: %s", sitemap_url, exc)

    return sorted(found)


def discover_internal_links(html: str, current_url: str) -> List[str]:
    """Extract same-domain relevant links from a page."""
    soup = BeautifulSoup(html or "", "html.parser")
    discovered = set()

    for anchor in soup.find_all("a", href=True):
        normalized = normalize_url(urljoin(current_url, anchor["href"]))
        if normalized and is_relevant_url(normalized):
            discovered.add(normalized)

    return sorted(discovered)


def build_url_queue() -> List[tuple]:
    """
    Build URL queue from seed pages, sitemap and internal links.
    Category is inferred from URL path and can be refined by the LLM.
    """
    queue = {}
    for url, category in SEED_URLS:
        normalized = normalize_url(url)
        if normalized:
            queue[normalized] = category

    for url in fetch_sitemap_urls():
        queue.setdefault(url, infer_category_from_url(url))

    # Sitemap can be incomplete. Inspect seed pages for internal links.
    scraper = cloudscraper.create_scraper()
    for seed_url, _category in SEED_URLS:
        try:
            response = scraper.get(seed_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                for link in discover_internal_links(response.text, seed_url):
                    queue.setdefault(link, infer_category_from_url(link))
        except Exception as exc:
            logger.warning("Could not discover links from %s: %s", seed_url, exc)

    urls = list(queue.items())[:MAX_PAGES]
    logger.info("Total URLs queued: %d (MAX_PAGES=%d)", len(urls), MAX_PAGES)
    return urls


def infer_category_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/").split("/")[-1]
    if not path:
        return "Company Information"

    label = path.replace("-", " ").replace("_", " ").title()
    return CANONICAL_CATEGORIES.get(label.lower(), label)


# ============================================================
# STANDARDIZATION / GOVERNANCE
# ============================================================

def standardize_category(raw_category: str) -> str:
    key = (raw_category or "").strip().lower()
    return CANONICAL_CATEGORIES.get(key, (raw_category or "General Information").strip())


def standardize_terminology(text: str) -> str:
    for pattern, replacement in TERMINOLOGY_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def standardize_headings(text: str) -> str:
    """Normalize markdown headings while preserving their hierarchy."""
    lines = text.splitlines()
    output = []
    has_h1 = False

    for line in lines:
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            output.append(line)
            continue

        hashes, heading = match.groups()
        heading = re.sub(r"\s+", " ", heading).strip()
        level = len(hashes)

        if not has_h1:
            level = 1
            has_h1 = True
        else:
            level = max(level, 2)

        output.append(f"{'#' * level} {heading}")

    return "\n".join(output)


def mask_pii(text: str, findings: Optional[list] = None) -> str:
    findings = findings if findings is not None else []

    for kind, pattern in PII_PATTERNS.items():
        def replace(match, pii_kind=kind):
            findings.append({"type": pii_kind, "masked_value": "[REDACTED]"})
            return f"[REDACTED_{pii_kind.upper()}]"
        text = pattern.sub(replace, text)

    return text


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def is_near_duplicate(text: str, accepted_texts: list,
                      threshold: float = NEAR_DUPLICATE_THRESHOLD) -> bool:
    normalized = normalize_for_hash(text)
    if not normalized:
        return False

    # Avoid expensive similarity checks for very large collections.
    for existing in accepted_texts:
        if abs(len(normalized) - len(existing)) / max(len(normalized), len(existing), 1) > 0.15:
            continue
        if difflib.SequenceMatcher(None, normalized, existing).ratio() >= threshold:
            return True

    return False


def clean_scraped_markdown(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", text)

    # Remove common repeated navigation / footer lines.
    boilerplate_exact = {
        "home", "about us", "contact us", "login", "sign up",
        "privacy policy", "terms & conditions", "terms and conditions",
        "facebook", "instagram", "linkedin", "youtube",
    }

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue

        if stripped.lower() in boilerplate_exact:
            continue

        if re.fullmatch(r"[-|•\s]+", stripped):
            continue

        lines.append(stripped)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = standardize_headings(text)
    text = standardize_terminology(text)
    return text.strip()


def apply_standardization_pipeline(raw_text: str, raw_category: str):
    text = clean_scraped_markdown(raw_text)
    pii_findings = []
    text = mask_pii(text, pii_findings)
    category = standardize_category(raw_category)
    return text, category, pii_findings


# ============================================================
# SCRAPING CONFIGURATION
# ============================================================

def scrape_with_cloudscraper(url: str) -> tuple:
    """Fallback scraper. Returns (clean text, raw html)."""
    scraper = cloudscraper.create_scraper()
    try:
        response = scraper.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup([
                "script", "style", "nav", "footer", "header", "aside",
                "noscript", "svg",
            ]):
                element.decompose()

            # Remove common cookie / social / navigation containers.
            for selector in [
                "[class*='cookie']", "[id*='cookie']", "[class*='navbar']",
                "[class*='navigation']", "[class*='footer']", "[class*='header']",
            ]:
                for element in soup.select(selector):
                    element.decompose()

            return soup.get_text(separator="\n", strip=True), response.text

        logger.warning("Fallback HTTP status %s for %s", response.status_code, url)

    except Exception as exc:
        logger.warning("Cloudscraper error for %s: %s", url, exc)

    return "", ""


JS_EXPAND_FAQS = r"""
document.querySelectorAll(
  'button[aria-expanded="false"], .accordion-header, [class*="faq"], [class*="question"], summary'
).forEach(el => {
  try { el.click(); } catch(e) {}
});
document.querySelectorAll('details').forEach(el => el.setAttribute('open', 'true'));
"""


BROWSER_CONFIG = BrowserConfig(
    browser_type="chromium",
    headless=True,
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    },
)


CRAWLER_CONFIG = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(),
    excluded_tags=[
        "nav", "footer", "header", "aside", "script", "style",
        "noscript", "svg",
    ],
    excluded_selector=(
        "footer, header, nav, aside, "
        "[class*='footer'], [class*='navbar'], [class*='navigation'], "
        "[class*='cookie'], [id*='cookie']"
    ),
    js_code=JS_EXPAND_FAQS,
    delay_before_return_html=3.0,
    wait_until="networkidle",
    page_timeout=45000,
    cache_mode=CacheMode.BYPASS,
)


async def scrape_one_page(crawler, url: str) -> dict:
    """Scrape one URL with Crawl4AI, then fallback if needed."""
    markdown = ""
    html = ""
    method = "crawl4ai"
    title = ""

    try:
        result = await crawler.arun(url=url, config=CRAWLER_CONFIG)

        if result.success:
            html = getattr(result, "html", "") or ""
            metadata = getattr(result, "metadata", {}) or {}
            title = metadata.get("title", "") if isinstance(metadata, dict) else ""

            markdown_obj = getattr(result, "markdown", None)
            if markdown_obj:
                markdown = (
                    getattr(markdown_obj, "fit_markdown", None)
                    or getattr(markdown_obj, "raw_markdown", None)
                    or (markdown_obj if isinstance(markdown_obj, str) else "")
                )

    except Exception as exc:
        logger.warning("Crawl4AI failed for %s: %s", url, exc)

    cleaned = clean_scraped_markdown(markdown)

    if len(cleaned) < MIN_CONTENT_LENGTH:
        logger.info("Using Cloudscraper fallback for %s", url)
        fallback_text, fallback_html = scrape_with_cloudscraper(url)
        if len(fallback_text) > len(cleaned):
            cleaned = clean_scraped_markdown(fallback_text)
            html = fallback_html
            method = "cloudscraper"

    if not title and html:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

    return {
        "url": url,
        "title": title or infer_category_from_url(url),
        "text": cleaned,
        "html": html,
        "method": method,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


async def run_scraper(url_queue: List[tuple]) -> list:
    """Scrape queued pages and discover links one level deeper."""
    scraped_pages = []
    discovered_queue = list(url_queue)
    known_urls = {url for url, _ in discovered_queue}

    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        index = 0
        while index < len(discovered_queue) and len(scraped_pages) < MAX_PAGES:
            url, category = discovered_queue[index]
            index += 1

            logger.info("Scraping %d/%d: %s", index, min(len(discovered_queue), MAX_PAGES), url)
            page = await scrape_one_page(crawler, url)

            if len(page["text"]) < MIN_CONTENT_LENGTH:
                logger.warning("Skipping page with insufficient text: %s", url)
                continue

            page["category"] = category
            scraped_pages.append(page)

            # Discover relevant internal links from rendered HTML.
            for link in discover_internal_links(page.get("html", ""), url):
                if link not in known_urls and len(discovered_queue) < MAX_PAGES:
                    known_urls.add(link)
                    discovered_queue.append((link, infer_category_from_url(link)))

            await asyncio.sleep(CRAWL_DELAY_SECONDS)

    return scraped_pages


# ============================================================
# GROQ + PYDANTIC STRUCTURED EXTRACTION
# ============================================================

class FAQItem(BaseModel):
    question: str = Field(description="Question explicitly found on the source page.")
    answer: str = Field(description="Answer supported by the source page.")


class KeyFact(BaseModel):
    topic: str = Field(description="Short fact topic.")
    fact: str = Field(description="Fact stated on the source page.")


class PageKnowledge(BaseModel):
    source_url: str
    page_title: str
    page_type: str = Field(
        description="One of: company, service, consultation, contact, policy, faq, other."
    )
    summary: str = ""
    services_mentioned: List[str] = Field(default_factory=list)
    who_is_it_for: List[str] = Field(default_factory=list)
    features_and_benefits: List[str] = Field(default_factory=list)
    booking_steps: List[str] = Field(default_factory=list)
    pricing_information: List[str] = Field(default_factory=list)
    faqs: List[FAQItem] = Field(default_factory=list)
    contact_information: List[str] = Field(default_factory=list)
    policy_points: List[str] = Field(default_factory=list)
    key_facts: List[KeyFact] = Field(default_factory=list)
    information_not_found: List[str] = Field(default_factory=list)


_token_logs: dict = {}      # model -> list of (monotonic timestamp, tokens)
_dead_models: set = set()   # models that returned "not found" (skipped afterwards)

JSON_TEMPLATE = """{
  "source_url": "<exact source url>",
  "page_title": "<title>",
  "page_type": "company | service | consultation | contact | policy | faq | other",
  "summary": "<2-3 sentence summary>",
  "services_mentioned": ["..."],
  "who_is_it_for": ["..."],
  "features_and_benefits": ["..."],
  "booking_steps": ["..."],
  "pricing_information": ["..."],
  "faqs": [{"question": "...", "answer": "..."}],
  "contact_information": ["..."],
  "policy_points": ["..."],
  "key_facts": [{"topic": "...", "fact": "..."}],
  "information_not_found": ["..."]
}"""


def _tpm_limit(model: str) -> int:
    return MODEL_TPM_LIMITS.get(model, DEFAULT_TPM_LIMIT)


def _tokens_used_last_minute(model: str) -> int:
    log = _token_logs.setdefault(model, [])
    now = time.monotonic()
    while log and now - log[0][0] > 60:
        log.pop(0)
    return sum(tokens for _, tokens in log)


async def _wait_for_token_budget(model: str, estimated: int):
    """Simple tokens-per-minute governor so we stay under Groq's TPM limit."""
    while True:
        used = _tokens_used_last_minute(model)
        log = _token_logs.setdefault(model, [])
        if not log or used + estimated <= _tpm_limit(model) * 0.85:
            return
        wait = 60 - (time.monotonic() - log[0][0]) + 0.5
        logger.info("[%s] ~%d tokens used in last 60s, waiting %.0fs", model, used, wait)
        await asyncio.sleep(max(wait, 1.0))


def _parse_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def _build_prompts(page: dict) -> tuple:
    system = (
        "You extract factual customer-support knowledge from company website "
        "content. You never invent services, prices, medical advice, availability, "
        "guarantees, or policies. Respond with ONE valid JSON object only."
    )
    user = f"""Extract customer-support knowledge from this The CarePal webpage and
return it as JSON matching exactly this structure:

{JSON_TEMPLATE}

SOURCE URL: {page['url']}
PAGE TITLE: {page.get('title', '')}
PAGE CATEGORY HINT: {page.get('category', '')}

PAGE CONTENT:
----------------
{page['text'][:MAX_LLM_CHARS]}
----------------

Rules:
1. Use ONLY facts explicitly supported by the page content.
2. Do not infer or invent pricing, caregiver availability, appointment slots,
   medical diagnoses, treatment recommendations, or booking status.
3. If a value is not present, use an empty list or empty string.
4. Extract FAQs only when a question and its answer are present in the content.
5. Keep booking steps in the order shown by the page.
6. Keep public business contact information if it is explicitly present.
7. Do not turn testimonials into company guarantees.
8. This is a customer-support knowledge base, not a source of medical advice.
9. Return source_url exactly as supplied.
10. If information is contradictory or unclear, mention it in information_not_found.
11. Output valid JSON only: all keys and string values in double quotes."""
    return system, user


async def _extract_with_model(page: dict, model: str, api_key: str) -> dict:
    """Try one Groq model (JSON mode + retries + TPM throttling)."""
    import litellm

    system, user = _build_prompts(page)
    estimated = int((len(system) + len(user)) / 3.5) + 1500  # prompt + output

    kwargs = dict(
        model=f"groq/{model}",
        api_key=api_key,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=3000,
        response_format={"type": "json_object"},
        timeout=90,          # never hang on a single call
    )
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"   # keep reasoning tokens small

    last_error = None
    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        await _wait_for_token_budget(model, estimated)
        try:
            started = time.monotonic()
            response = await litellm.acompletion(**kwargs)

            usage = getattr(response, "usage", None)
            total = getattr(usage, "total_tokens", None) or estimated
            _token_logs.setdefault(model, []).append((time.monotonic(), int(total)))

            data = _parse_json_object(response.choices[0].message.content)
            data = {k: v for k, v in data.items() if v is not None}
            data["source_url"] = page["url"]
            data.setdefault("page_title", page.get("title", ""))
            data.setdefault("page_type", "other")
            result = PageKnowledge.model_validate(data).model_dump()
            logger.info("[%s] OK in %.1fs (%d tokens): %s",
                        model, time.monotonic() - started, int(total), page["url"])
            return result

        except litellm.NotFoundError:
            raise   # model missing -> caller moves to the next model

        except litellm.RateLimitError as exc:
            last_error = exc
            match = re.search(r"try again in ([\d.]+)s", str(exc))
            wait = float(match.group(1)) + 1.5 if match else 20.0
            logger.warning("[%s] rate limited (attempt %d/%d), waiting %.0fs",
                           model, attempt, LLM_MAX_ATTEMPTS, wait)
            await asyncio.sleep(wait)

        except litellm.BadRequestError as exc:
            last_error = exc
            if "reasoning_effort" in kwargs and "reasoning" in str(exc).lower():
                kwargs.pop("reasoning_effort", None)
                continue
            if "response_format" in kwargs and "json" in str(exc).lower() \
                    and "tool_use_failed" not in str(exc):
                kwargs.pop("response_format", None)   # JSON mode unsupported
                continue
            logger.warning("[%s] bad request (attempt %d/%d): %s",
                           model, attempt, LLM_MAX_ATTEMPTS, str(exc)[:300])
            await asyncio.sleep(2)

        except ValueError as exc:   # JSON decode / Pydantic validation errors
            last_error = exc
            logger.warning("[%s] invalid JSON/schema (attempt %d/%d): %s",
                           model, attempt, LLM_MAX_ATTEMPTS, str(exc)[:300])
            await asyncio.sleep(2)

        except Exception as exc:    # network / timeout etc.
            last_error = exc
            logger.warning("[%s] call failed (attempt %d/%d): %s",
                           model, attempt, LLM_MAX_ATTEMPTS, str(exc)[:300])
            await asyncio.sleep(3)

    raise RuntimeError(f"{model}: failed after {LLM_MAX_ATTEMPTS} attempts: {last_error}")


async def extract_page_knowledge(page: dict) -> dict:
    """
    Structured extraction for one page. Tries each model in GROQ_MODELS in
    order and falls back to the next one on 'model not found' or repeated
    failure. Uses Groq JSON mode via litellm (instead of CrewAI tool-calling,
    which fails with tool_use_failed on several Groq models).
    """
    import litellm

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to environment variables or set USE_CREWAI=False."
        )

    errors = []
    for model in GROQ_MODELS:
        if model in _dead_models:
            continue
        try:
            return await _extract_with_model(page, model, api_key)
        except litellm.NotFoundError:
            _dead_models.add(model)
            logger.warning("Model not available, skipping from now on: %s", model)
            errors.append(f"{model}: not found")
        except Exception as exc:
            logger.warning("Model %s failed for %s, trying next model.", model, page["url"])
            errors.append(str(exc)[:300])

    raise RuntimeError("All models failed: " + " | ".join(errors))


async def structure_pages_with_crewai(pages: list) -> list:
    """Extract each page independently so one failure doesn't stop the pipeline."""
    structured_pages = []

    for index, page in enumerate(pages, start=1):
        logger.info("LLM extraction %d/%d: %s", index, len(pages), page["url"])

        try:
            structured = await extract_page_knowledge(page)
            structured["source_url"] = page["url"]
            structured["source_category"] = page["category"]
            structured["scraped_at"] = page["scraped_at"]
            structured["content_hash"] = content_hash(page["text"])
            structured_pages.append(structured)

        except Exception as exc:
            logger.error("Extraction failed for %s: %s", page["url"], str(exc)[:500])
            # Preserve raw page as a fallback record. Do not discard source data.
            structured_pages.append({
                "source_url": page["url"],
                "source_category": page["category"],
                "page_title": page.get("title", ""),
                "page_type": "unstructured_fallback",
                "summary": "",
                "raw_content": page["text"],
                "extraction_error": str(exc)[:1000],
                "scraped_at": page["scraped_at"],
                "content_hash": content_hash(page["text"]),
            })

        await asyncio.sleep(CREWAI_DELAY_SECONDS)

    return structured_pages


# ============================================================
# LANGCHAIN DOCUMENTS (one per page, no chunking)
# ============================================================

def structured_record_to_text(record: dict) -> str:
    """Convert structured JSON record into readable text."""
    parts = []

    def add(label, value):
        if value is None:
            return
        if isinstance(value, list):
            if not value:
                return
            if all(isinstance(item, dict) for item in value):
                rendered = []
                for item in value:
                    rendered.append(
                        " | ".join(f"{key}: {val}" for key, val in item.items() if val)
                    )
                value = "\n".join(rendered)
            else:
                value = "\n".join(str(item) for item in value if item)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)

        if value.strip():
            parts.append(f"{label}:\n{value.strip()}")

    add("Page title", record.get("page_title"))
    add("Page type", record.get("page_type"))
    add("Summary", record.get("summary"))
    add("Services mentioned", record.get("services_mentioned"))
    add("Who is it for", record.get("who_is_it_for"))
    add("Features and benefits", record.get("features_and_benefits"))
    add("Booking steps", record.get("booking_steps"))
    add("Pricing information", record.get("pricing_information"))
    add("Frequently asked questions", record.get("faqs"))
    add("Contact information", record.get("contact_information"))
    add("Policy points", record.get("policy_points"))
    add("Key facts", record.get("key_facts"))
    add("Information not found", record.get("information_not_found"))

    if record.get("raw_content"):
        add("Raw source content", record.get("raw_content"))

    return "\n\n".join(parts)


def build_langchain_documents(structured_pages: list) -> list:
    """One Document per page. No text splitting / chunking."""
    documents = []

    for index, record in enumerate(structured_pages, start=1):
        text = structured_record_to_text(record)
        if len(text.strip()) < 30:
            continue

        metadata = {
            "record_id": f"carepal_{index:04d}",
            "title": record.get("page_title", ""),
            "category": record.get("source_category", record.get("page_type", "General")),
            "source": record.get("source_url", ""),
            "source_type": "website",
            "content_hash": record.get("content_hash", ""),
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            if isinstance(record, Document):
                row = {
                    "page_content": record.page_content,
                    "metadata": record.metadata,
                }
            else:
                row = record
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_scrape_audit(pages: list):
    path = OUTPUT_DIR / "scrape_audit.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["url", "title", "category", "method", "characters", "content_hash"],
        )
        writer.writeheader()
        for page in pages:
            writer.writerow({
                "url": page["url"],
                "title": page.get("title", ""),
                "category": page.get("category", ""),
                "method": page.get("method", ""),
                "characters": len(page.get("text", "")),
                "content_hash": content_hash(page.get("text", "")),
            })


# ============================================================
# MAIN PIPELINE
# ============================================================

async def main():
    logger.info(
        "Starting The CarePal knowledge-base pipeline. [SCRIPT VERSION: v7 (no chunking) | models=%s]",
        ", ".join(GROQ_MODELS),
    )

    cached_file = OUTPUT_DIR / "cleaned_pages.json"
    if REUSE_CLEANED_PAGES and cached_file.exists():
        # Scraping already done in an earlier run: reuse it (saves minutes).
        accepted_pages = json.loads(cached_file.read_text(encoding="utf-8"))
        url_queue = [(p["url"], p["category"]) for p in accepted_pages]
        scraped_pages = accepted_pages
        duplicate_count = 0
        pii_audit = []
        logger.info("Reusing %d cleaned pages from %s (skipping scraping). "
                    "Delete that file or set REUSE_CLEANED_PAGES=False to re-scrape.",
                    len(accepted_pages), cached_file)
    else:
        # 1. Discover URLs.
        url_queue = build_url_queue()
        if not url_queue:
            logger.error("No URLs discovered. Check internet access / website availability.")
            return

        # 2. Scrape.
        scraped_pages = await run_scraper(url_queue)

        # 3. Exact + near-duplicate removal and standardization.
        accepted_pages = []
        seen_hashes = set()
        seen_texts = []
        duplicate_count = 0
        pii_audit = []

        for page in scraped_pages:
            cleaned, category, pii_findings = apply_standardization_pipeline(
                page["text"], page["category"]
            )

            if len(cleaned) < MIN_CONTENT_LENGTH:
                continue

            digest = content_hash(cleaned)

            if digest in seen_hashes or is_near_duplicate(cleaned, seen_texts):
                duplicate_count += 1
                logger.info("Duplicate skipped: %s", page["url"])
                continue

            page["text"] = cleaned
            page["category"] = category
            page["content_hash"] = digest

            if pii_findings:
                pii_audit.append({
                    "source_url": page["url"],
                    "finding_count": len(pii_findings),
                    "findings": pii_findings,
                })

            accepted_pages.append(page)
            seen_hashes.add(digest)
            seen_texts.append(normalize_for_hash(cleaned))

        logger.info(
            "Scraped=%d | Accepted=%d | Duplicates=%d | PII-masked pages=%d",
            len(scraped_pages), len(accepted_pages), duplicate_count, len(pii_audit)
        )

        # Save raw/cleaned source pages regardless of LLM result.
        save_json(OUTPUT_DIR / "cleaned_pages.json", [
            {
                "url": p["url"],
                "title": p.get("title", ""),
                "category": p["category"],
                "text": p["text"],
                "method": p.get("method"),
                "scraped_at": p.get("scraped_at"),
                "content_hash": p["content_hash"],
            }
            for p in accepted_pages
        ])
        save_scrape_audit(accepted_pages)
        save_json(OUTPUT_DIR / "pii_audit.json", pii_audit)

    # 4. Structured extraction.
    if USE_CREWAI:
        # Direct Groq JSON-mode extraction with model fallback + TPM throttling.
        structured_pages = await structure_pages_with_crewai(accepted_pages)
    else:
        structured_pages = [
            {
                "source_url": p["url"],
                "source_category": p["category"],
                "page_title": p.get("title", ""),
                "page_type": "unstructured_fallback",
                "raw_content": p["text"],
                "content_hash": p["content_hash"],
            }
            for p in accepted_pages
        ]

    save_json(OUTPUT_DIR / "structured_knowledge.json", structured_pages)

    # 5. Build LangChain documents (one per page, no chunking).
    documents = build_langchain_documents(structured_pages)
    save_jsonl(OUTPUT_DIR / "langchain_documents.jsonl", documents)

    # 6. Summary.
    summary = {
        "website": BASE_URL,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "urls_queued": len(url_queue),
        "pages_scraped": len(scraped_pages),
        "pages_accepted": len(accepted_pages),
        "duplicates_skipped": duplicate_count,
        "structured_records": len(structured_pages),
        "documents": len(documents),
        "pii_masked_pages": len(pii_audit),
        "crewai_enabled": USE_CREWAI,
        "output_directory": str(OUTPUT_DIR.resolve()),
    }

    save_json(OUTPUT_DIR / "run_summary.json", summary)

    logger.info("Pipeline completed.")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")