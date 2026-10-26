"""Heading/paragraph-aware chunking for Markdown knowledge articles."""
from __future__ import annotations

import re
from pathlib import Path

from src.config import KB_DIR


def load_documents() -> list[dict[str, str]]:
    documents = []
    for path in sorted(KB_DIR.rglob("*.md")):
        if path.name.startswith("DRAFT_"):
            continue
        documents.append({
            "doc_name": path.name,
            "relative_path": str(path.relative_to(KB_DIR)),
            "text": path.read_text(encoding="utf-8"),
        })
    return documents


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        additional = len(word) + (1 if current else 0)
        if current and current_len + additional > max_chars:
            chunk_text = " ".join(current).strip()
            chunks.append(chunk_text)
            # Word-based overlap avoids chopping words in half.
            carry: list[str] = []
            carry_len = 0
            for old_word in reversed(current):
                add_len = len(old_word) + (1 if carry else 0)
                if carry_len + add_len > overlap:
                    break
                carry.insert(0, old_word)
                carry_len += add_len
            current = carry
            current_len = len(" ".join(current))
            if current and current_len + len(word) + 1 > max_chars:
                current = []
                current_len = 0
        current.append(word)
        current_len += additional if len(current) > 1 else len(word)
    if current:
        chunks.append(" ".join(current).strip())
    return [x for x in chunks if x]


def create_chunks(documents: list[dict[str, str]], max_chars: int = 800, overlap: int = 100) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for doc in documents:
        text = doc["text"].strip()
        # Preserve Markdown headings as section boundaries.
        sections = re.split(r"(?m)(?=^#{1,6}\s+)", text)
        for section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            heading_match = re.search(r"(?m)^#{1,6}\s+(.+)$", section_text)
            heading = heading_match.group(1).strip() if heading_match else "General"
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
            current = ""
            for paragraph in paragraphs:
                if len(paragraph) > max_chars:
                    if current:
                        chunks.append({"chunk_id": len(chunks), "doc_name": doc["doc_name"], "section": heading, "text": current})
                        current = ""
                    for part in _split_long_text(paragraph, max_chars, overlap):
                        chunks.append({"chunk_id": len(chunks), "doc_name": doc["doc_name"], "section": heading, "text": part})
                    continue
                candidate = f"{current}\n\n{paragraph}" if current else paragraph
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append({"chunk_id": len(chunks), "doc_name": doc["doc_name"], "section": heading, "text": current})
                    current = paragraph
            if current.strip():
                chunks.append({"chunk_id": len(chunks), "doc_name": doc["doc_name"], "section": heading, "text": current.strip()})
    return chunks
