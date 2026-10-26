"""Budget-aware Gemini wrapper. Tracks usage metadata for every API attempt."""
from __future__ import annotations

import math
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types

from src.config import (
    API_MIN_INTERVAL_SECONDS,
    API_USAGE_LOG,
    DAILY_TOKEN_BUDGET,
    LAST_API_CALL_FILE,
    MAX_DAILY_API_CALLS,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    EXPERIMENT_LABEL,
)
from src.telemetry import log_api_usage, read_jsonl, today_records


class UsageBudgetExceeded(RuntimeError):
    """Raised before a request when the local usage budget would be exceeded."""


class APIKeyMissing(RuntimeError):
    """Raised when GEMINI_API_KEY is not set."""


def estimate_tokens_from_chars(text: str) -> int:
    """Conservative rough estimate only; exact counts come from response metadata."""
    return max(1, math.ceil(len(text) / 3.0))


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_gemini_api_key":
        raise APIKeyMissing("Add your Gemini API key to the .env file as GEMINI_API_KEY=...")
    return genai.Client(api_key=api_key)


def _current_day_api_records() -> list[dict[str, Any]]:
    return today_records(read_jsonl(API_USAGE_LOG))


def _check_local_budget(prompt: str, output_limit: int) -> dict[str, int]:
    todays = _current_day_api_records()
    call_count = len(todays)  # Includes failed requests, so repeated failures are bounded too.
    actual_tokens = sum(
        int(r.get("total_tokens") or 0)
        for r in todays if r.get("status") == "success"
    )
    estimated_input = estimate_tokens_from_chars(prompt)
    estimated_request = estimated_input + output_limit

    if call_count >= MAX_DAILY_API_CALLS:
        raise UsageBudgetExceeded(
            f"Local request budget reached ({MAX_DAILY_API_CALLS} API attempts today). "
            "Increase MAX_DAILY_API_CALLS in .env only if appropriate."
        )
    if actual_tokens + estimated_request > DAILY_TOKEN_BUDGET:
        raise UsageBudgetExceeded(
            f"Local token budget guard stopped the request. Used {actual_tokens} reported tokens today; "
            f"roughly {estimated_request} more could be needed, against a {DAILY_TOKEN_BUDGET}-token budget."
        )
    return {"requests_today_before_call": call_count, "reported_tokens_today_before_call": actual_tokens,
            "estimated_input_tokens": estimated_input, "estimated_request_tokens": estimated_request}


def _wait_for_rate_limit() -> None:
    """Persist request pacing between app reruns. A local limiter cannot coordinate other devices."""
    if API_MIN_INTERVAL_SECONDS <= 0 or not LAST_API_CALL_FILE.exists():
        return
    try:
        last = float(LAST_API_CALL_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return
    wait = API_MIN_INTERVAL_SECONDS - (time.time() - last)
    if wait > 0:
        time.sleep(wait)


def _usage_value(metadata: Any, field: str) -> int:
    return int(getattr(metadata, field, 0) or 0) if metadata is not None else 0


def generate_text(prompt: str, purpose: str, question_label: str, max_output_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
    """Call Gemini once, enforce local budgets, and record response usage metadata."""
    output_limit = max_output_tokens or MAX_OUTPUT_TOKENS
    budget = _check_local_budget(prompt, output_limit)
    _wait_for_rate_limit()
    LAST_API_CALL_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_API_CALL_FILE.write_text(str(time.time()), encoding="utf-8")
    started = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")

    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=output_limit,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
            ),
        ),
        )
        latency = round(time.perf_counter() - started, 3)
        metadata = getattr(response, "usage_metadata", None)
        input_tokens = _usage_value(metadata, "prompt_token_count")
        output_tokens = _usage_value(metadata, "candidates_token_count")
        thought_tokens = _usage_value(metadata, "thoughts_token_count")
        cached_tokens = _usage_value(metadata, "cached_content_token_count")
        tool_tokens = _usage_value(metadata, "tool_use_prompt_token_count")
        total_tokens = _usage_value(metadata, "total_token_count")
        if not total_tokens:
            # Fallback for responses without the total field; reported categories may still be zero.
            total_tokens = input_tokens + output_tokens + thought_tokens

        record = {
            "timestamp": timestamp,
            "purpose": purpose,
            "experiment": EXPERIMENT_LABEL,
            "question_label": question_label[:160],
            "model": MODEL_NAME,
            "status": "success",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thought_tokens,
            "cached_tokens": cached_tokens,
            "tool_use_prompt_tokens": tool_tokens,
            "total_tokens": total_tokens,
            "estimated_input_tokens_before_call": budget["estimated_input_tokens"],
            "output_token_limit": output_limit,
            "latency_seconds": latency,
            "prompt_chars": len(prompt),
            "error": "",
        }
        log_api_usage(record)
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text, record
    except errors.APIError as exc:
        latency = round(time.perf_counter() - started, 3)
        log_api_usage({
            "timestamp": timestamp,
            "purpose": purpose,
            "experiment": EXPERIMENT_LABEL,
            "question_label": question_label[:160],
            "model": MODEL_NAME,
            "status": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
            "tool_use_prompt_tokens": 0,
            "total_tokens": 0,
            "estimated_input_tokens_before_call": budget["estimated_input_tokens"],
            "output_token_limit": output_limit,
            "latency_seconds": latency,
            "prompt_chars": len(prompt),
            "error": f"{getattr(exc, 'code', '')}: {getattr(exc, 'message', str(exc))}",
        })
        raise
