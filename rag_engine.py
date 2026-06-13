"""
HR RAG Engine
Vector search using TF-IDF + cosine similarity (stdlib + numpy only)
Ollama REST API for LLM inference
"""

import json
import re
import math
import numpy as np
from pathlib import Path


# ── Text utilities ──────────────────────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "but", "not", "this", "that", "these", "those", "it", "its",
    "i", "you", "we", "they", "he", "she", "their", "our", "your",
    "what", "how", "when", "where", "which", "who", "my", "me"
}


def tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ── TF-IDF Vector Store ─────────────────────────────────────────────────────────

class TFIDFVectorStore:
    """
    In-memory TF-IDF vector store.
    Replaces ChromaDB / FAISS for the prototype — no external deps needed.
    """

    def __init__(self):
        self.documents: list[dict] = []      # raw docs
        self.vocab: list[str] = []           # sorted unique terms
        self.idf: np.ndarray | None = None
        self.doc_vecs: np.ndarray | None = None  # shape (n_docs, vocab_size)

    # ── Build ──────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[dict]) -> None:
        """
        docs: list of {"id", "title", "category", "content", "chunk"}
        """
        self.documents = docs
        tokenised = [tokenise(d["chunk"]) for d in docs]

        # Vocabulary
        vocab_set: set[str] = set()
        for toks in tokenised:
            vocab_set.update(toks)
        self.vocab = sorted(vocab_set)
        word_idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        N = len(docs)

        # TF matrix
        tf = np.zeros((N, V), dtype=np.float32)
        for i, toks in enumerate(tokenised):
            for tok in toks:
                if tok in word_idx:
                    tf[i, word_idx[tok]] += 1
            row_sum = tf[i].sum()
            if row_sum > 0:
                tf[i] /= row_sum  # term frequency (normalised)

        # IDF
        df = (tf > 0).sum(axis=0).astype(np.float32)  # document frequency
        self.idf = np.log((N + 1) / (df + 1)) + 1.0   # smooth IDF

        # TF-IDF
        self.doc_vecs = tf * self.idf

        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(self.doc_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.doc_vecs = self.doc_vecs / norms

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 3) -> list[dict]:
        """Return top_k most relevant documents with similarity scores."""
        if not self.documents or self.doc_vecs is None:
            return []

        word_idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        qvec = np.zeros(V, dtype=np.float32)
        for tok in tokenise(text):
            if tok in word_idx:
                qvec[word_idx[tok]] += 1

        # apply IDF
        qvec = qvec * self.idf
        norm = np.linalg.norm(qvec)
        if norm == 0:
            return []
        qvec /= norm

        scores = self.doc_vecs @ qvec  # cosine similarities
        top_idxs = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idxs:
            if scores[idx] > 0.01:   # relevance threshold
                results.append({
                    **self.documents[idx],
                    "score": float(scores[idx])
                })
        return results


# ── Document loader ─────────────────────────────────────────────────────────────

def load_policies(data_path: str = "data/hr_policies.json") -> TFIDFVectorStore:
    """Load HR policies JSON into the vector store."""
    path = Path(data_path)
    with path.open() as f:
        data = json.load(f)

    docs = []
    for pol in data["policies"]:
        # Chunk = title + content for better retrieval signal
        chunk = f"{pol['title']}. Category: {pol['category']}. {pol['content']}"
        docs.append({
            "id": pol["id"],
            "title": pol["title"],
            "category": pol["category"],
            "content": pol["content"],
            "chunk": chunk,
        })

    store = TFIDFVectorStore()
    store.add_documents(docs)
    return store


# ── Ollama LLM call ─────────────────────────────────────────────────────────────

import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"   # change to any model you have pulled


def call_ollama(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    stream: bool = False,
) -> str:
    """
    Call Ollama /api/chat endpoint.
    Returns the assistant text content.
    Raises OllamaError with a friendly message on failure.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": 0.2, "num_predict": 600}
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
            return body["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running (`ollama serve`) and "
            f"the model is pulled (`ollama pull {model}`). "
            f"Details: {e}"
        )
    except KeyError:
        raise OllamaError("Unexpected response format from Ollama.")


class OllamaError(Exception):
    pass


# ── RAG pipeline ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are HRBot, a helpful and professional HR assistant for employees.
Your job is to answer HR-related questions accurately based ONLY on the company policy documents provided.

Rules:
- Answer directly and concisely.
- If the answer is in the context, provide it clearly and mention the policy name.
- If the answer is NOT in the provided context, say: "I don't have specific information about that in our policy documents. Please contact the HR team directly."
- Never make up numbers, dates, or rules not present in the context.
- Use a friendly, professional tone.
- Format numbers and lists clearly when needed.
"""


def answer_question(
    question: str,
    store: TFIDFVectorStore,
    chat_history: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    top_k: int = 3,
) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant policy chunks
      2. Build prompt with context + chat history
      3. Call Ollama
      4. Return answer + sources
    """
    # 1. Retrieve
    hits = store.query(question, top_k=top_k)
    context_blocks = []
    sources = []
    for h in hits:
        context_blocks.append(
            f"[Policy: {h['title']} | Category: {h['category']}]\n{h['content']}"
        )
        sources.append({"id": h["id"], "title": h["title"], "category": h["category"], "score": h["score"]})

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant policy found."

    # 2. Build messages
    user_content = (
        f"Context from HR policy documents:\n{context}\n\n"
        f"Employee question: {question}"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])   # last 3 turns
    messages.append({"role": "user", "content": user_content})

    # 3. Call LLM
    answer = call_ollama(messages, model=model)

    return {
        "answer": answer,
        "sources": sources,
        "context_used": context_blocks,
    }


def list_ollama_models() -> list[str]:
    """Return available Ollama models, or [] on error."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
