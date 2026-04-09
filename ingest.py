"""
Ingest crawled CSV data into ChromaDB for RAG retrieval.

Usage: python3 ingest.py <file.csv> [file2.csv ...]
       python3 ingest.py *.csv

Reads CSV files from the crawler, chunks content by headings,
embeds via Ollama (nomic-embed-text), stores in ChromaDB.
"""

import csv
import json
import sys
import re
import time
from pathlib import Path
import requests
import chromadb
from tqdm import tqdm

csv.field_size_limit(10 * 1024 * 1024)  # 10MB max field size

# Config

CHROMADB_URL   = "http://localhost:8000"
OLLAMA_URL     = "http://localhost:11434"
EMBED_MODEL    = "nomic-embed-text"
COLLECTION     = "webcrawler"
CHUNK_MAX_WORDS = 300  # split chunks larger than this


# Embed text via Ollama

def embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    """Embed a batch of texts via Ollama with retry."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  WARNING:Embed failed ({e}), retrying {attempt + 2}/{retries}...", flush=True)
                time.sleep(2)
            else:
                raise


# Split content into chunks by heading boundaries

def chunk_by_headings(content: str, headings_json: str | None) -> list[dict]:
    """Split content into chunks using heading boundaries.
    Returns list of {section, text} dicts."""
    if not content or not content.strip():
        return []

    # If no headings, just split by word count
    if not headings_json:
        return _split_long_text(content, "main")

    try:
        headings = json.loads(headings_json)
    except (json.JSONDecodeError, TypeError):
        return _split_long_text(content, "main")

    if not headings:
        return _split_long_text(content, "main")

    # Try to split content at heading boundaries
    chunks = []
    heading_texts = [h["text"] for h in headings]

    # Build sections by finding heading positions in content
    sections = []
    remaining = content
    for h_text in heading_texts:
        idx = remaining.find(h_text)
        if idx > 0:
            # Text before this heading belongs to previous section
            before = remaining[:idx].strip()
            if before:
                sections.append(before)
            remaining = remaining[idx:]
        elif idx == 0:
            remaining = remaining[len(h_text):]

    if remaining.strip():
        sections.append(remaining.strip())

    if not sections:
        return _split_long_text(content, "main")

    # Pair sections with heading names
    for i, section_text in enumerate(sections):
        section_name = heading_texts[i] if i < len(heading_texts) else "continued"
        chunks.extend(_split_long_text(section_text, section_name))

    return chunks


def _split_long_text(text: str, section: str) -> list[dict]:
    """Split text into chunks of CHUNK_MAX_WORDS words."""
    words = text.split()
    if len(words) <= CHUNK_MAX_WORDS:
        return [{"section": section, "text": text.strip()}] if text.strip() else []

    chunks = []
    for i in range(0, len(words), CHUNK_MAX_WORDS):
        chunk_text = " ".join(words[i:i + CHUNK_MAX_WORDS])
        part = f" (part {i // CHUNK_MAX_WORDS + 1})" if len(words) > CHUNK_MAX_WORDS else ""
        chunks.append({"section": f"{section}{part}", "text": chunk_text})
    return chunks


# CSV reader

def read_csv(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


# Ingest a CSV file into ChromaDB

def ingest_file(collection, filepath: str) -> int:
    """Ingest one CSV file into ChromaDB. Returns number of chunks added."""
    path = Path(filepath)
    if not path.exists():
        print(f"SKIP (file not found)")
        return 0
    if not path.suffix == ".csv":
        print(f"SKIP (not a .csv file)")
        return 0

    rows = read_csv(filepath)
    if not rows:
        print(f"SKIP (empty)")
        return 0

    if "content" not in rows[0]:
        print(f"SKIP (missing 'content' column)")
        return 0

    total = 0
    skipped = 0

    # First pass: collect all chunks
    all_ids = []
    all_docs = []
    all_meta = []

    for row in rows:
        url = row.get("url", "")
        content = row.get("content", "")
        error = row.get("error", "")

        if error or not content.strip():
            skipped += 1
            continue

        title = row.get("title", "")
        domain = row.get("domain", "")
        headings = row.get("headings")
        content_type = row.get("content_type", "")
        date = row.get("date", "")

        chunks = chunk_by_headings(content, headings)
        for i, chunk in enumerate(chunks):
            doc_text = f"{title} - {chunk['section']}: {chunk['text']}"
            chunk_id = re.sub(r"[^\w]", "_", f"{url}_{i}")
            all_ids.append(chunk_id)
            all_docs.append(doc_text)
            all_meta.append({
                "url": url,
                "domain": domain,
                "title": title,
                "section": chunk["section"],
                "content_type": content_type,
                "date": date or "",
            })

    if not all_docs:
        if skipped:
            print(f"  ({skipped} rows skipped — empty/errored)", flush=True)
        return 0

    print(f"\n  {len(all_docs)} chunks from {len(rows) - skipped} pages", flush=True)
    if skipped:
        print(f"  ({skipped} rows skipped — empty/errored)", flush=True)

    # Second pass: embed and upsert with progress bar
    pbar = tqdm(total=len(all_docs), desc="  Embedding", unit="chunk", ncols=80)
    for b in range(0, len(all_docs), 32):
        batch_ids = all_ids[b:b+32]
        batch_docs = all_docs[b:b+32]
        batch_meta = all_meta[b:b+32]
        try:
            batch_embeds = embed(batch_docs)
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeds,
                metadatas=batch_meta,
            )
            total += len(batch_ids)
        except Exception as e:
            tqdm.write(f"  WARNING:Batch failed: {e}")
        pbar.update(len(batch_ids))
    pbar.close()

    return total


# Entry point

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ingest.py <file.csv> [file2.csv ...]")
        sys.exit(1)

    files = sys.argv[1:]

    # Pre-flight
    print("Checking Ollama ...", end=" ", flush=True)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        assert r.status_code == 200
        models = [m["name"] for m in r.json().get("models", [])]
        print("OK")
    except Exception:
        print("FAIL")
        print(f"  Ollama not responding at {OLLAMA_URL}")
        print("  Run: docker start ollama")
        sys.exit(1)

    print(f"Checking model '{EMBED_MODEL}' ...", end=" ", flush=True)
    if any(EMBED_MODEL in m for m in models):
        print("OK")
    else:
        print("NOT FOUND")
        print(f"  Run: docker exec ollama ollama pull {EMBED_MODEL}")
        sys.exit(1)

    print("Checking ChromaDB ...", end=" ", flush=True)
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
        print("OK")
    except Exception:
        print("FAIL")
        print(f"  ChromaDB not responding at {CHROMADB_URL}")
        print("  Run: docker start chromadb")
        sys.exit(1)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()
    print(f"\nCollection '{COLLECTION}': {existing} existing chunks\n")

    # Ingest each file
    total = 0
    for filepath in files:
        print(f"Ingesting {filepath} ...", end=" ", flush=True)
        try:
            n = ingest_file(collection, filepath)
            print(f"{n} chunks")
            total += n
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n+ Done. {total} chunks added. Collection now has {collection.count()} total.")


if __name__ == "__main__":
    main()
