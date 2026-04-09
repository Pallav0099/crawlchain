"""
SearXNG + Headless Firefox -> ML-Ready Structured CSV
------------------------------------------------------
Usage: python3 main.py
  1. Enter a URL
  2. Headless Firefox renders JS, extracts clean structured data
  3. SearXNG finds related pages, scrapes those too
  4. Saves to <domain>.csv — clean enough for ML training

Requires:
  docker compose up -d          (searxng)
  playwright install firefox    (headless gecko)
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
NUM_RESULTS  = 10
DELAY_SEC    = 1.5
PAGE_TIMEOUT = 20000  # ms


# Data model for extracted page info

@dataclass
class PageData:
    url:            str
    domain:         str
    title:          str
    content:        str            # main body text, boilerplate stripped
    description:    str
    author:         Optional[str] = None
    date:           Optional[str] = None
    language:       Optional[str] = None
    content_type:   Optional[str] = None  # article, product, homepage, etc.
    word_count:     int = 0
    headings:       Optional[str] = None  # JSON list of h1-h3
    links_internal: int = 0
    links_external: int = 0
    image_count:    int = 0
    schema_types:   Optional[str] = None  # JSON-LD @types found
    error:          Optional[str] = None


# Headless Firefox page fetcher

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


# SearXNG search

def search(query: str, num_results: int) -> list[dict]:
    params = {
        "q":        query,
        "format":   "json",
        "language": "en",
        "pageno":   1,
    }
    resp = requests.get(SEARXNG_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", [])[:num_results]


# Page extraction using trafilatura + BeautifulSoup

def extract_page(url: str, html: str) -> PageData:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    soup = BeautifulSoup(html, "html.parser")

    # Clean main content via trafilatura
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_recall=True,
        url=url,
        output_format="txt",
    )
    content = _clean_text(extracted or "")

    # Metadata via trafilatura
    metadata = trafilatura.extract_metadata(html, default_url=url)

    title = ""
    author = None
    date = None
    description = ""
    language = None

    if metadata:
        title = metadata.title or ""
        author = metadata.author or None
        date = metadata.date or None
        description = _clean_text(metadata.description or "")
        language = metadata.language or None

    # Fallback title from HTML
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Headings (h1-h3)
    headings = []
    for level in ["h1", "h2", "h3"]:
        for tag in soup.find_all(level):
            text = tag.get_text(strip=True)
            if text:
                headings.append({"level": level, "text": text})
    headings_json = json.dumps(headings, ensure_ascii=False) if headings else None

    # Links
    links_int = 0
    links_ext = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("http://", "https://")):
            if domain in href:
                links_int += 1
            else:
                links_ext += 1
        elif href.startswith("/"):
            links_int += 1

    # Images
    image_count = len(soup.find_all("img"))

    # Schema.org types
    schema_types = _extract_schema_types(soup)

    # Content type heuristic
    content_type = _guess_content_type(soup, schema_types, content)

    # Word count
    word_count = len(content.split()) if content else 0

    return PageData(
        url=url,
        domain=domain,
        title=_clean_text(title),
        content=content,
        description=description,
        author=author,
        date=date,
        language=language,
        content_type=content_type,
        word_count=word_count,
        headings=headings_json,
        links_internal=links_int,
        links_external=links_ext,
        image_count=image_count,
        schema_types=schema_types,
    )


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip artifacts for ML-clean text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # Remove zero-width and control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]", "", text)
    return text


def _extract_schema_types(soup: BeautifulSoup) -> Optional[str]:
    types = set()
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue
        _collect_types(data, types)
    return ",".join(sorted(types)) if types else None


def _collect_types(data, types: set):
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


def _guess_content_type(soup, schema_types: Optional[str], content: str) -> str:
    if schema_types:
        st = schema_types.lower()
        if "product" in st:
            return "product"
        if "article" in st or "newsarticle" in st or "blogposting" in st:
            return "article"
        if "recipe" in st:
            return "recipe"
        if "faqpage" in st:
            return "faq"

    # Heuristic from HTML
    if soup.find("article"):
        return "article"
    if soup.find(attrs={"itemprop": "price"}):
        return "product"

    word_count = len(content.split()) if content else 0
    if word_count < 50:
        return "thin"
    if word_count > 500:
        return "article"
    return "page"


# CSV output

def save_csv(rows: list[PageData], filepath: str):
    column_names = [f.name for f in fields(PageData)]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        for row in rows:
            writer.writerow(astuple(row))
    print(f"\n+ Saved {len(rows)} rows -> {filepath}")


# Helpers

def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path
    name = name.strip("/")
    name = re.sub(r"[^\w.\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > 120:
        name = name[:120]
    return name + ".csv"


def check_searxng() -> bool:
    try:
        r = requests.get(SEARXNG_URL, params={"q": "test", "format": "json"}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def check_firefox() -> bool:
    try:
        with sync_playwright() as pw:
            browser = pw.firefox.launch(headless=True)
            browser.close()
        return True
    except Exception:
        return False


def scrape_single(browser: Browser, url: str) -> PageData:
    print(f"  Fetching {url} ...")
    html = fetch_html(browser, url)
    if html is None:
        return PageData(url=url, domain=urlparse(url).netloc,
                        title="", content="", description="", error="fetch_failed")
    page_data = extract_page(url, html)
    print(f"  + [{page_data.content_type}] {page_data.word_count} words, "
          f"{page_data.links_internal}i/{page_data.links_external}e links")
    return page_data


def scrape_via_searxng(browser: Browser, query: str) -> list[PageData]:
    print(f"\nQuerying SearXNG: '{query}'")
    results = search(query, NUM_RESULTS)
    print(f"   Found {len(results)} results\n")

    rows: list[PageData] = []
    for i, result in enumerate(results, 1):
        url = result.get("url", "")
        print(f"[{i}/{len(results)}] {url}")
        try:
            rows.append(scrape_single(browser, url))
        except Exception as e:
            rows.append(PageData(url=url, domain=urlparse(url).netloc,
                                 title="", content="", description="", error=str(e)))
        if i < len(results):
            time.sleep(DELAY_SEC)
    return rows


# Entry point

def main():
    print("Checking SearXNG ...", end=" ", flush=True)
    if not check_searxng():
        print("FAIL")
        print(f"  WARNING: SearXNG not responding at {SEARXNG_URL}")
        print("     Run: docker compose up -d")
        sys.exit(1)
    print("OK")

    print("Checking Firefox ...", end=" ", flush=True)
    if not check_firefox():
        print("FAIL")
        print("  WARNING: Headless Firefox could not launch")
        print("     Run: playwright install firefox")
        sys.exit(1)
    print("OK")

    try:
        url = input("\nenter url: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(0)

    if not url:
        print("No URL provided.")
        sys.exit(1)

    print("\nStarting headless Firefox ...")
    with sync_playwright() as pw:
        browser = pw.firefox.launch(headless=True)

        try:
            print(f"\nDirect scrape")
            try:
                direct = scrape_single(browser, url)
            except Exception as e:
                direct = PageData(url=url, domain=urlparse(url).netloc,
                                  title="", content="", description="", error=str(e))

            print(f"\nSearXNG search")
            search_rows = scrape_via_searxng(browser, url)

            rows = [direct]
            seen = {url}
            for row in search_rows:
                if row.url not in seen:
                    rows.append(row)
                    seen.add(row.url)

            outfile = url_to_filename(url)
            save_csv(rows, outfile)
            print(f"   {outfile}")

        finally:
            browser.close()
            print("Browser closed.")


if __name__ == "__main__":
    main()
