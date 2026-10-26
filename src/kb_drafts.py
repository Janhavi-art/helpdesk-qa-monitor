"""Draft a knowledge article from aggregate ticket themes; drafts are never auto-indexed."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.config import KB_DRAFTS_DIR, MAX_KB_DRAFT_TOKENS
from src.llm import generate_text


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value[:70] or "topic"


def draft_knowledge_article(category: str, topic: str, keywords: list[dict[str, Any]]) -> tuple[str, Path, dict[str, Any]]:
    """Generate a short unverified draft using only aggregate phrases, not raw tickets."""
    phrases = [str(item.get("term", "")) for item in keywords[:12] if item.get("term")]
    keywords_text = ", ".join(phrases) if phrases else "No recurring phrases found"
    prompt = f"""Draft a short IT knowledge-base article for human review.

Ticket category: {category}
Topic selected by the maintainer: {topic}
Frequent aggregate phrases from ticket descriptions (not resolutions): {keywords_text}

Important constraints:
- These ticket phrases describe user-reported issues; they do not establish a verified fix.
- Do not invent organization-specific portals, URLs, permissions, internal processes, or policy.
- Give only conservative, general steps that are broadly safe.
- Where a company-specific step or admin permission is needed, explicitly say to consult official IT instructions or contact IT.
- Include: symptoms, safe checks, steps, when to escalate, and what information to include in a support request.
- Keep the article under 500 words.
- Start the article with a visible warning: 'DRAFT — UNVERIFIED; review against authoritative documentation before publishing.'
- Do not claim that the article has been reviewed or is official.
Return Markdown only.
"""
    content, usage = generate_text(
        prompt=prompt,
        purpose="knowledge_article_draft",
        question_label=f"{category}: {topic}",
        max_output_tokens=MAX_KB_DRAFT_TOKENS,
    )
    KB_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = KB_DRAFTS_DIR / f"DRAFT_{_slug(category)}_{_slug(topic)}.md"
    destination.write_text(content.rstrip() + "\n", encoding="utf-8")
    return content, destination, usage
