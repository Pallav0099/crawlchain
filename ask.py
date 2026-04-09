"""
RAG query interface — ask questions about crawled data.

Usage: python3 ask.py
       python3 ask.py "What is skytunnel?"

Searches ChromaDB for relevant chunks, sends them as context
to phi4-mini via Ollama for grounded answers.
"""

import json
import sys
import time
import requests
import chromadb
from tqdm import tqdm

# Config

CHROMADB_URL  = "http://localhost:8000"
OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
CHAT_MODEL    = "phi4-mini:3.8b-q4_K_M"
COLLECTION    = "crawlchain"
TOP_K         = 5

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on retrieved web data.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so.
Cite the source URL when referencing specific information."""


# Embed query and search ChromaDB

def search(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and search ChromaDB for relevant chunks."""
    # Embed the question
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    query_embedding = resp.json()["embeddings"][0]

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "url": results["metadatas"][0][i].get("url", ""),
            "title": results["metadatas"][0][i].get("title", ""),
            "section": results["metadatas"][0][i].get("section", ""),
            "distance": results["distances"][0][i],
        })
    return hits


# Send question + context to LLM and stream response

def ask_llm(question: str, context_chunks: list[dict]) -> str:
    """Send question + context to phi4-mini and stream the response."""
    # Build context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['url']}\n"
            f"    Section: {chunk['section']}\n"
            f"    {chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    user_msg = f"""Context from crawled web data:

{context}

Question: {question}"""

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    # Stream tokens to stdout with stats
    full = []
    token_count = 0
    start = time.time()
    try:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full.append(token)
                    token_count += 1
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"\n  WARNING:Stream interrupted: {e}")
    except KeyboardInterrupt:
        print("\n  (interrupted)")

    elapsed = time.time() - start
    tok_s = token_count / elapsed if elapsed > 0 else 0
    print(f"\n  [{token_count} tokens, {elapsed:.1f}s, {tok_s:.1f} tok/s]")
    return "".join(full)


# Entry point

def main():
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

    for model in [EMBED_MODEL, CHAT_MODEL]:
        print(f"Checking model '{model}' ...", end=" ", flush=True)
        if any(model in m for m in models):
            print("OK")
        else:
            print("NOT FOUND")
            print(f"  Run: docker exec ollama ollama pull {model}")
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

    try:
        collection = client.get_collection(name=COLLECTION)
    except Exception:
        print(f"Collection '{COLLECTION}' not found.")
        print("Run: python3 ingest.py wiki_dataset.csv")
        sys.exit(1)

    count = collection.count()
    print(f"Collection '{COLLECTION}': {count} chunks\n")

    if count == 0:
        print("No data ingested yet. Run: python3 ingest.py *.csv")
        sys.exit(1)

    # One-shot mode
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Q: {question}\n")
        print("Searching ...", end=" ", flush=True)
        hits = search(collection, question)
        print(f"{len(hits)} chunks (best: {hits[0]['distance']:.3f})\n")
        ask_llm(question, hits)
        print(f"\n\nSources:")
        seen = set()
        for h in hits:
            if h["url"] not in seen:
                print(f"  {h['url']}")
                seen.add(h["url"])
        return

    # Interactive mode
    print("Ask anything (Ctrl+C to quit):\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue

        try:
            hits = search(collection, question)
            print(f"  ({len(hits)} chunks, best: {hits[0]['distance']:.3f})\n")
            ask_llm(question, hits)
            print(f"\nSources:")
            seen = set()
            for h in hits:
                if h["url"] not in seen:
                    print(f"  {h['url']}")
                    seen.add(h["url"])
            print()
        except Exception as e:
            print(f"  WARNING:Error: {e}\n")


if __name__ == "__main__":
    main()
