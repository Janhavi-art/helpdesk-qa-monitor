from pathlib import Path
import re


KB_PATH = Path("data/kb_docs")

CHUNK_SIZE = 700
OVERLAP = 100


def load_documents():
    """Read all Markdown documents from the knowledge base."""

    documents = []

    for file_path in KB_PATH.rglob("*.md"):

        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "doc_name": file_path.name,
            "path": str(file_path),
            "text": text
        })

    return documents


def split_into_chunks(text):
    """
    Split text into chunks of approximately CHUNK_SIZE characters
    with some overlap between consecutive chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - OVERLAP

    return chunks


def get_section(chunk):
    """
    Find the first Markdown heading in a chunk.
    """

    match = re.search(r"^#+\s+(.+)$", chunk, re.MULTILINE)

    if match:
        return match.group(1).strip()

    return "General"


def create_chunks(documents):
    """Create chunks and attach metadata."""

    all_chunks = []

    chunk_id = 0

    for document in documents:

        chunks = split_into_chunks(document["text"])

        for chunk in chunks:

            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_name": document["doc_name"],
                "section": get_section(chunk),
                "text": chunk
            })

            chunk_id += 1

    return all_chunks


if __name__ == "__main__":

    documents = load_documents()

    print(f"Found {len(documents)} documents.")

    chunks = create_chunks(documents)

    print(f"\nCreated {len(chunks)} chunks.\n")

    for chunk in chunks:

        print("=" * 60)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Document: {chunk['doc_name']}")
        print(f"Section: {chunk['section']}")
        print(f"Characters: {len(chunk['text'])}")
        print(chunk["text"][:200])
        print()