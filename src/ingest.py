"""Build or rebuild the local ChromaDB knowledge index."""
from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, CACHE_FILE
from src.chunking import load_documents, create_chunks


def rebuild_index() -> dict[str, int]:
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Markdown knowledge articles found under {CHROMA_DIR.parent / 'data' / 'kb_docs'}")
    chunks = create_chunks(documents)
    if not chunks:
        raise RuntimeError("Knowledge articles were found, but no text chunks were created.")

    print(f"Documents found: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print("Loading local embedding model (first run may download it)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [str(c["text"]) for c in chunks]
    print("Embedding knowledge chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_names = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in collection_names:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['doc_name']}::{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"doc_name": str(c["doc_name"]), "section": str(c["section"])}
        for c in chunks
    ]
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {collection.count()} chunks in {CHROMA_DIR}.")
    # A changed knowledge base invalidates cached answers.
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    return {"documents": len(documents), "chunks": len(chunks)}


def ensure_index() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_names = [c.name if hasattr(c, "name") else str(c) for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME not in collection_names:
        rebuild_index()
        return
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    if collection.count() == 0:
        rebuild_index()


if __name__ == "__main__":
    rebuild_index()
