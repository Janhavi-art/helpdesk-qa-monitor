"""Chunk knowledge-base Markdown articles and store vectors in persistent ChromaDB."""
import re
import chromadb
from sentence_transformers import SentenceTransformer
from config import KB_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def split_article(text: str) -> list[str]:
    """Split an article into heading/paragraph chunks, preserving each useful block."""
    chunks = []
    current_heading = ""
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
        if paragraph:
            # Keep a subsection heading with the paragraph it introduces.
            chunk_text = f"{current_heading}\n{paragraph}".strip()
            chunks.append(chunk_text)
        paragraph_lines = []

    for line in text.splitlines():
        if line.startswith("# "):
            # The article title is handled separately and should not become a standalone chunk.
            continue
        if line.startswith("## "):
            flush_paragraph()
            current_heading = line[3:].strip()
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph_lines.append(line)
    flush_paragraph()
    return chunks


def load_articles() -> list[dict]:
    """Read articles and derive section, title, and article ID from the files."""
    articles = []
    for path in sorted(KB_DIR.glob("*/*.md")):
        text = path.read_text(encoding="utf-8")
        first_line = next((line.strip() for line in text.splitlines() if line.startswith("# ")), "")
        title = first_line[2:].strip() if first_line else path.stem
        article_id = path.stem
        section_slug = path.parent.name
        section = {
            "wifi": "Wi-Fi", "passwords": "Passwords", "software": "Software",
            "hardware": "Hardware", "network_access": "Network access"
        }.get(section_slug, section_slug.replace("_", " ").title())
        for index, chunk in enumerate(split_article(text), start=1):
            # Prepending the article title gives the embedding model useful context.
            chunk_with_title = f"Article title: {title}\nSection: {section}\n{chunk}"
            articles.append({
                "chunk_id": f"{article_id}-chunk-{index}",
                "text": chunk_with_title,
                "section": section,
                "article_title": title,
                "article_id": article_id,
                "source_file": path.name,
                "chunk_index": index,
            })
    if not articles:
        raise RuntimeError(f"No Markdown articles found under {KB_DIR}.")
    return articles


def build_index() -> int:
    """Embed every chunk and rebuild the persistent cosine-distance collection."""
    chunks = load_articles()
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Delete/recreate to prevent outdated chunks remaining after an article is edited.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[{
            "section": chunk["section"],
            "article_title": chunk["article_title"],
            "article_id": chunk["article_id"],
            "source_file": chunk["source_file"],
            "chunk_index": chunk["chunk_index"],
        } for chunk in chunks],
    )
    print(f"Indexed {len(chunks)} chunks from {len(list(KB_DIR.glob('*/*.md')))} articles.")
    print(f"Persistent ChromaDB directory: {CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    build_index()
