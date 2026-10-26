"""Run the small evaluation set. Results resume across runs; delete output to start fresh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import EVAL_SET_FILE, EVAL_RESULTS_FILE, PROCESSED_DIR, EXPERIMENT_LABEL
from src.rag import answer_question, clear_response_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HelpDeskQA Monitor retrieval and abstention.")
    parser.add_argument("--fresh", action="store_true", help="Clear existing results for this experiment first.")
    parser.add_argument("--no-cache", action="store_true", help="Do not use the answer cache for this run.")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET_FILE.read_text(encoding="utf-8"))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and EVAL_RESULTS_FILE.exists():
        EVAL_RESULTS_FILE.unlink()
    if args.fresh and args.no_cache:
        clear_response_cache()

    done: set[str] = set()
    if EVAL_RESULTS_FILE.exists():
        try:
            old = pd.read_csv(EVAL_RESULTS_FILE)
            done = set(old.loc[old["status"] == "completed", "question"].astype(str))
        except (ValueError, KeyError, OSError):
            done = set()

    for idx, case in enumerate(cases, start=1):
        question = str(case["question"])
        if question in done:
            print(f"[{idx}/{len(cases)}] Already completed for '{EXPERIMENT_LABEL}'; skipping: {question}")
            continue
        print(f"\n[{idx}/{len(cases)}] {question}")
        rec = answer_question(question, use_cache=not args.no_cache)
        sources = [s.get("doc_name") for s in rec.get("sources", [])]
        gold_doc = case.get("gold_doc")
        answerable = bool(case.get("answerable"))
        retrieval_hit = (gold_doc in sources) if gold_doc else None
        finished_status = rec.get("status") in {"success", "cache_hit", "local_abstention", "local_safety_refusal"}
        abstention_correct = (bool(rec.get("abstained")) == (not answerable)) if finished_status else None

        row = {
            "experiment": EXPERIMENT_LABEL,
            "question": question,
            "answerable": answerable,
            "gold_doc": gold_doc or "",
            "retrieved_sources": "; ".join(str(s) for s in sources),
            "retrieval_hit": retrieval_hit,
            "abstained": bool(rec.get("abstained")),
            "abstention_correct": abstention_correct,
            "cache_hit": bool(rec.get("cache_hit")),
            "status": "completed" if finished_status else str(rec.get("status")),
            "total_tokens": int(rec.get("tokens_this_request", 0) or 0),
            "latency_seconds": float(rec.get("latency_seconds", 0) or 0),
            "top_score": float(rec.get("top_score", 0) or 0),
            "answer": rec.get("answer", ""),
        }
        # Append immediately, so an API quota failure does not lose completed cases.
        existing = pd.read_csv(EVAL_RESULTS_FILE) if EVAL_RESULTS_FILE.exists() else pd.DataFrame()
        if not existing.empty and "question" in existing.columns:
            existing = existing[existing["question"].astype(str) != question]
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(EVAL_RESULTS_FILE, index=False)

    results = pd.read_csv(EVAL_RESULTS_FILE) if EVAL_RESULTS_FILE.exists() else pd.DataFrame()
    if results.empty:
        print("No evaluation results yet.")
        return

    print("\n=== Evaluation summary ===")
    print(f"Experiment: {EXPERIMENT_LABEL}")
    print(f"Cases recorded: {len(results)}")
    completed = results[results["status"] == "completed"]
    answerable_results = completed[completed["answerable"] == True]
    valid_hits = answerable_results["retrieval_hit"].dropna()
    if len(valid_hits):
        hits = valid_hits.astype(str).str.lower().isin(["true", "1"])
        print(f"Retrieval hit rate on answerable cases: {hits.mean() * 100:.1f}%")
    if not completed.empty:
        valid_abstention = completed["abstention_correct"].dropna()
        if not valid_abstention.empty:
            correct = valid_abstention.astype(str).str.lower().isin(["true", "1"])
            print(f"Correct abstention rate: {correct.mean() * 100:.1f}%")
    print(f"Failed or blocked cases: {(results['status'] != 'completed').sum()}")
    print(f"Gemini tokens reported in saved results: {pd.to_numeric(results['total_tokens'], errors='coerce').fillna(0).sum():.0f}")
    print(f"Saved to: {EVAL_RESULTS_FILE}")
    print("Note: automated metrics check retrieval/abstention, not factual answer correctness. Manually review sample answers.")


if __name__ == "__main__":
    main()
