"""
Batch crawl Wikipedia pages via SearXNG + Headless Firefox.

Usage: python3 batch_crawl.py
Outputs: wiki_dataset.csv (all pages combined)
"""

import csv
import json
import re
import sys
import time
import requests
import trafilatura
from bs4 import BeautifulSoup
from dataclasses import dataclass, fields, astuple
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, Browser

# Config

SEARXNG_URL  = "http://localhost:4545/search"
PAGE_TIMEOUT = 20000
DELAY_SEC    = 2.0
OUTPUT_FILE  = "wiki_dataset.csv"
TARGET_COUNT = 100

TOPICS = [
    # AI & ML
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "transformer architecture",
    "natural language processing",
    "generative adversarial network",
    "reinforcement learning",
    "convolutional neural network",
    "recurrent neural network",
    "GPT language model",
    "BERT language model",
    "diffusion model AI",
    "retrieval augmented generation",
    "vector database",
    "word embedding",
    "attention mechanism neural network",
    "fine-tuning machine learning",
    "federated learning",

    # Developer tools & practices
    "software development",
    "version control git",
    "continuous integration",
    "DevOps",
    "containerization Docker",
    "Kubernetes",
    "microservices architecture",
    "REST API",
    "GraphQL",
    "WebAssembly",
    "open source software",
    "Linux kernel",
    "compiler design",
    "programming language theory",
    "garbage collection computer science",

    # LLMs & AI systems
    "OpenAI",
    "ChatGPT",
    "Claude AI Anthropic",
    "Google DeepMind",
    "Llama language model Meta",
    "AI alignment",
    "AI safety",
    "prompt engineering",
    "AI hallucination",
    "mixture of experts",
    "knowledge distillation",
    "quantization neural networks",
    "RLHF reinforcement learning human feedback",

    # Computer networks & protocols
    "computer network",
    "TCP IP protocol",
    "HTTP protocol",
    "DNS domain name system",
    "BGP routing protocol",
    "TLS transport layer security",
    "WebSocket protocol",
    "QUIC protocol",
    "IPv6",
    "network address translation NAT",
    "VPN virtual private network",
    "SSH protocol",
    "WireGuard",
    "peer-to-peer network",
    "content delivery network",
    "reverse proxy",
    "load balancing computing",

    # Cryptography & identity
    "public key cryptography",
    "RSA cryptosystem",
    "elliptic curve cryptography",
    "zero knowledge proof",
    "digital signature",
    "hash function cryptography",
    "AES encryption",
    "blockchain technology",
    "X.509 certificate",
    "OAuth protocol",
    "WebAuthn",
    "FIDO2 authentication",
    "homomorphic encryption",
    "post-quantum cryptography",

    # Homelab & self-hosting
    "home server",
    "self-hosting internet services",
    "Raspberry Pi",
    "network attached storage",
    "virtualization technology",
    "Proxmox virtual environment",
    "reverse proxy server Nginx",
    "Pi-hole DNS",
    "WireGuard VPN setup",

    # Recent breakthroughs
    "AlphaFold protein structure",
    "neuromorphic computing",
    "quantum computing",
    "RISC-V processor",
    "solid state battery",
    "brain computer interface",
    "autonomous vehicle technology",
    "nuclear fusion energy",
    "CRISPR gene editing",
    "6G wireless technology",
    "optical computing",
    "memristor",
]


# Page extraction (same logic as main.py)

@dataclass
class PageData:
    url:            str
    domain:         str
    title:          str
    content:        str
    description:    str
    author:         Optional[str] = None
    date:           Optional[str] = None
    language:       Optional[str] = None
    content_type:   Optional[str] = None
    word_count:     int = 0
    headings:       Optional[str] = None
    links_internal: int = 0
    links_external: int = 0
    image_count:    int = 0
    schema_types:   Optional[str] = None
    error:          Optional[str] = None


def fetch_html(browser: Browser, url: str) -> Optional[str]:
    page = browser.new_page()
    try:
        page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT)
        time.sleep(1)
        return page.content()
    except Exception:
        return None
    finally:
        page.close()


def extract_page(url: str, html: str) -> PageData:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    soup = BeautifulSoup(html, "html.parser")

    extracted = trafilatura.extract(
        html, include_comments=False, include_tables=True,
        no_fallback=False, favor_recall=True, url=url, output_format="txt",
    )
    content = _clean_text(extracted or "")

    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = author = date = description = language = ""
    if metadata:
        title = metadata.title or ""
        author = metadata.author or None
        date = metadata.date or None
        description = _clean_text(metadata.description or "")
        language = metadata.language or None

    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    headings = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            text = tag.get_text(strip=True)
            if text:
                headings.append({"level": level, "text": text})
    headings_json = json.dumps(headings, ensure_ascii=False) if headings else None

    links_int = links_ext = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("http://", "https://")):
            if domain in href:
                links_int += 1
            else:
                links_ext += 1
        elif href.startswith("/"):
            links_int += 1

    image_count = len(soup.find_all("img"))
    schema_types = _extract_schema_types(soup)
    content_type = "article" if content and len(content.split()) > 200 else "page"
    word_count = len(content.split()) if content else 0

    return PageData(
        url=url, domain=domain, title=_clean_text(title), content=content,
        description=description, author=author, date=date, language=language,
        content_type=content_type, word_count=word_count, headings=headings_json,
        links_internal=links_int, links_external=links_ext,
        image_count=image_count, schema_types=schema_types,
    )


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]", "", text)
    return text


def _extract_schema_types(soup):
    types = set()
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue
        _collect_types(data, types)
    return ",".join(sorted(types)) if types else None


def _collect_types(data, types):
    if isinstance(data, list):
        for item in data:
            _collect_types(item, types)
    elif isinstance(data, dict):
        t = data.get("@type")
        if t:
            if isinstance(t, list):
                types.update(t)
            else:
                types.add(t)
        if "@graph" in data:
            _collect_types(data["@graph"], types)


# SearXNG Wikipedia search

def search_wiki(query: str) -> list[str]:
    """Search SearXNG and filter for Wikipedia URLs."""
    params = {
        "q": f"{query} wikipedia",
        "format": "json",
        "language": "en",
        "pageno": 1,
    }
    try:
        resp = requests.get(SEARXNG_URL, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        urls = []
        for r in results:
            url = r.get("url", "")
            # Only keep English Wikipedia article pages (no Talk:, User:, etc.)
            if "en.wikipedia.org/wiki/" in url:
                slug = url.split("/wiki/")[-1]
                if ":" not in slug and slug not in ("Main_Page",):
                    urls.append(url.split("#")[0])  # strip anchors
        return urls
    except Exception:
        return []


# Entry point

def main():
    print("Checking SearXNG ...", end=" ", flush=True)
    try:
        r = requests.get(SEARXNG_URL, params={"q": "test", "format": "json"}, timeout=5)
        assert r.status_code == 200
        print("OK")
    except Exception:
        print("FAIL — run: docker compose up -d")
        sys.exit(1)

    # Phase 1: collect Wikipedia URLs via SearXNG
    print(f"\nPhase 1: Collecting Wikipedia URLs ({len(TOPICS)} topics)\n")
    collected = {}  # url -> topic
    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i}/{len(TOPICS)}] Searching: {topic} ...", end=" ", flush=True)
        urls = search_wiki(topic)
        new = 0
        for url in urls:
            if url not in collected:
                collected[url] = topic
                new += 1
        print(f"{len(urls)} results, {new} new (total: {len(collected)})")

        if len(collected) >= TARGET_COUNT:
            print(f"\n  Reached {TARGET_COUNT} URLs, stopping search.")
            break
        time.sleep(1)  # polite delay for SearXNG

    # Trim to target
    urls = list(collected.keys())[:TARGET_COUNT]
    print(f"\n  Collected {len(urls)} unique Wikipedia URLs\n")

    # Phase 2: crawl each URL
    print(f"Phase 2: Crawling {len(urls)} pages\n")
    print("Starting headless Firefox ...")

    rows = []
    with sync_playwright() as pw:
        browser = pw.firefox.launch(headless=True)
        try:
            for i, url in enumerate(urls, 1):
                print(f"[{i}/{len(urls)}] {url.split('/wiki/')[-1][:50]} ...", end=" ", flush=True)
                html = fetch_html(browser, url)
                if html is None:
                    rows.append(PageData(url=url, domain="en.wikipedia.org",
                                         title="", content="", description="",
                                         error="fetch_failed"))
                    print("FAILED")
                    continue

                try:
                    page = extract_page(url, html)
                    rows.append(page)
                    print(f"{page.word_count} words")
                except Exception as e:
                    rows.append(PageData(url=url, domain="en.wikipedia.org",
                                         title="", content="", description="",
                                         error=str(e)))
                    print(f"ERROR: {e}")

                if i < len(urls):
                    time.sleep(DELAY_SEC)
        finally:
            browser.close()

    # Phase 3: save
    column_names = [f.name for f in fields(PageData)]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        for row in rows:
            writer.writerow(astuple(row))

    ok = sum(1 for r in rows if not r.error)
    total_words = sum(r.word_count for r in rows)
    print(f"\n+ Done. {ok}/{len(rows)} pages scraped, {total_words:,} total words")
    print(f"  Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
