"""
RAG API server — serves the web UI.

Usage: python3 server.py
Endpoints:
  POST /ask    — { "question": "..." } -> streamed answer
  GET  /stats  — collection info
"""

import csv
import json
import sys
import time
import requests
import chromadb
from flask import Flask, request, Response, jsonify
from flask_cors import CORS

# Config

CHROMADB_URL  = "http://localhost:8000"
OLLAMA_URL    = "http://localhost:11434"
EMBED_MODEL   = "nomic-embed-text"
CHAT_MODEL    = "phi4-mini:3.8b-q4_K_M"
COLLECTION    = "crawlchain"
TOP_K         = 5
PORT          = 5000

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on retrieved web data.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so.
Cite the source URL when referencing specific information."""

csv.field_size_limit(10 * 1024 * 1024)

app = Flask(__name__)
CORS(app)

# ChromaDB client (lazy init)

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.HttpClient(host="localhost", port=8000)
        _collection = _client.get_collection(name=COLLECTION)
    return _collection


# Embed query and search ChromaDB

def search(query: str, top_k: int = TOP_K) -> list[dict]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    query_embedding = resp.json()["embeddings"][0]

    collection = get_collection()
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


# API routes

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or not data.get("question"):
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        hits = search(question)
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

    # Build context
    context_parts = []
    for i, chunk in enumerate(hits, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['url']}\n"
            f"    Section: {chunk['section']}\n"
            f"    {chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    user_msg = f"""Context from crawled web data:

{context}

Question: {question}"""

    def generate():
        # Send sources first
        sources = []
        seen = set()
        for h in hits:
            if h["url"] not in seen:
                sources.append({"url": h["url"], "title": h["title"], "distance": h["distance"]})
                seen.add(h["url"])
        yield json.dumps({"type": "sources", "data": sources}) + "\n"

        # Stream LLM response
        try:
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

            token_count = 0
            start = time.time()

            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        token_count += 1
                        yield json.dumps({"type": "token", "data": token}) + "\n"

            elapsed = time.time() - start
            tok_s = token_count / elapsed if elapsed > 0 else 0
            yield json.dumps({
                "type": "done",
                "tokens": token_count,
                "elapsed": round(elapsed, 1),
                "tok_s": round(tok_s, 1),
            }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return Response(generate(), mimetype="text/plain")


@app.route("/stats", methods=["GET"])
def stats():
    try:
        collection = get_collection()
        count = collection.count()

        # Check Ollama
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]

        return jsonify({
            "chunks": count,
            "collection": COLLECTION,
            "embed_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
            "ollama": "online",
            "chromadb": "online",
            "models_loaded": models,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Entry point

if __name__ == "__main__":
    # Pre-flight
    print("Checking Ollama ...", end=" ", flush=True)
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        assert r.status_code == 200
        print("OK")
    except Exception:
        print("FAIL — run: docker start ollama")
        sys.exit(1)

    print("Checking ChromaDB ...", end=" ", flush=True)
    try:
        c = chromadb.HttpClient(host="localhost", port=8000)
        c.heartbeat()
        col = c.get_collection(name=COLLECTION)
        print(f"OK ({col.count()} chunks)")
    except Exception:
        print("FAIL — run: docker start chromadb")
        sys.exit(1)

    print(f"\nServer running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
