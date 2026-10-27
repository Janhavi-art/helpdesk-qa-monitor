"""Simple retrieval and grounded answer generation for the HelpDeskQA project."""
import json
from functools import lru_cache
from google import genai
from google.genai import types
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GEMINI_API_KEY,
    GEMINI_MODEL, DEFAULT_THRESHOLD, TOP_K, THRESHOLD_FILE,
)

NOT_FOUND = "I couldn’t find this."


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-embedding model once per Python process."""
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Open the already-built persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("ChromaDB index is missing. Run `python -m src.ingest` first.") from exc


def get_threshold() -> float:
    """Use the tuned threshold when present; otherwise use the .env default."""
    if THRESHOLD_FILE.exists():
        try:
            value = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
            return float(value["similarity_threshold"])
        except (ValueError, KeyError, OSError, TypeError):
            pass
    return DEFAULT_THRESHOLD


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the most similar chunks with cosine similarity and article metadata."""
    question = question.strip()
    if not question:
        return []
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    vector = get_embedding_model().encode([question], normalize_embeddings=True)[0].tolist()
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(max(1, top_k), count),
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        # Chroma cosine distance = 1 - cosine similarity for this collection.
        output.append({
            "text": document,
            "section": metadata["section"],
            "article_title": metadata["article_title"],
            "article_id": metadata["article_id"],
            "source_file": metadata["source_file"],
            "similarity": round(1.0 - float(distance), 4),
        })
    return output


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Ask Gemini to answer only from retrieved context and return used article IDs."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "replace_with_your_key":
        raise ValueError("Add your Gemini API key to the .env file before asking questions.")

    allowed_ids = sorted({chunk["article_id"] for chunk in chunks})
    context = "\n\n".join(
        f"[ARTICLE_ID: {chunk['article_id']}] [TITLE: {chunk['article_title']}] "
        f"[SECTION: {chunk['section']}]\n{chunk['text']}"
        for chunk in chunks
    )
    prompt = f"""You are a careful IT helpdesk assistant.
Use only facts and steps stated in the retrieved context. Do not invent steps,
company policies, links, or permissions. If context does not answer the question,
say the available knowledge base does not contain enough information.
Return one valid JSON object with exactly these keys:
- "answer": a concise answer in plain text
- "article_ids_used": a list of IDs from the allowed IDs below that directly support the answer
Do not include any ID that is not in the allowed list. If no retrieved article supports
the answer, use an empty list for article_ids_used.

Allowed article IDs: {allowed_ids}

RETRIEVED CONTEXT:
{context}

QUESTION: {question}"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    try:
        payload = json.loads(response.text or "{}")
        answer = str(payload.get("answer", "")).strip()
        used = payload.get("article_ids_used", [])
        if not isinstance(used, list):
            used = []
        used_ids = [item for item in used if isinstance(item, str) and item in allowed_ids]
        if not answer:
            raise ValueError("Missing answer in Gemini JSON")
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        # Safe fallback: display the model text but cite only actually retrieved articles.
        answer = (response.text or "").strip()
        used_ids = allowed_ids
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
    return {"answer": answer, "article_ids_used": used_ids}


def answer_question(question: str, threshold: float | None = None) -> dict:
    """Retrieve, reject weak matches, then generate a grounded answer."""
    active_threshold = get_threshold() if threshold is None else threshold
    chunks = retrieve(question, top_k=TOP_K)
    if not chunks or chunks[0]["similarity"] < active_threshold:
        return {"answer": NOT_FOUND, "citations": [], "retrieved": chunks,
                "best_similarity": chunks[0]["similarity"] if chunks else None,
                "threshold": active_threshold}

    # Keep only chunks that clear the same confidence threshold.
    accepted = [chunk for chunk in chunks if chunk["similarity"] >= active_threshold]
    result = generate_answer(question, accepted)
    by_id = {chunk["article_id"]: chunk for chunk in accepted}
    citations = [by_id[article_id] for article_id in result["article_ids_used"] if article_id in by_id]
    return {"answer": result["answer"], "citations": citations, "retrieved": accepted,
            "best_similarity": chunks[0]["similarity"], "threshold": active_threshold}
