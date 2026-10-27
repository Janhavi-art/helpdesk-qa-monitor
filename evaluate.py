"""Evaluate retrieval (no Gemini calls) and tune the rejection threshold."""
import json
from pathlib import Path
from src.config import THRESHOLD_FILE
from src.ingest import build_index
from src.rag import retrieve

# 28 in-scope questions with an expected relevant article ID; four out-of-scope questions.
# These are starter evaluation examples. Add independently written questions before claiming
# production-level performance.
TEST_CASES = [
    {"question": "My Wi-Fi will not connect to the office network.", "expected": "KB-WIFI-001"},
    {"question": "The wireless connection keeps dropping every few minutes.", "expected": "KB-WIFI-002"},
    {"question": "I am connected but webpages do not load.", "expected": "KB-WIFI-003"},
    {"question": "The company Wi-Fi does not appear in the network list.", "expected": "KB-WIFI-004"},
    {"question": "It says my wireless password is incorrect.", "expected": "KB-WIFI-005"},
    {"question": "I forgot my company login password.", "expected": "KB-PASS-001"},
    {"question": "My work account was locked after sign-in attempts.", "expected": "KB-PASS-002"},
    {"question": "The password portal will not accept my new password.", "expected": "KB-PASS-003"},
    {"question": "My authenticator code is not being accepted.", "expected": "KB-PASS-004"},
    {"question": "How should I protect my work account credentials?", "expected": "KB-PASS-005"},
    {"question": "When I click the app, nothing opens.", "expected": "KB-SOFT-001"},
    {"question": "My desktop application has become very slow.", "expected": "KB-SOFT-002"},
    {"question": "What is the process to install a new work application?", "expected": "KB-SOFT-003"},
    {"question": "The application update stops with an error.", "expected": "KB-SOFT-004"},
    {"question": "I get a permissions error in a software tool.", "expected": "KB-SOFT-005"},
    {"question": "My laptop is completely dead when I press the power button.", "expected": "KB-HW-001"},
    {"question": "Some keys on my keyboard do not respond.", "expected": "KB-HW-002"},
    {"question": "The touchpad and mouse pointer are not working.", "expected": "KB-HW-003"},
    {"question": "My external screen says no signal.", "expected": "KB-HW-004"},
    {"question": "The laptop is very hot and its fan is loud.", "expected": "KB-HW-005"},
    {"question": "I cannot open a folder shared by my team.", "expected": "KB-NET-001"},
    {"question": "The company VPN connection keeps failing.", "expected": "KB-NET-002"},
    {"question": "An internal company website is not loading.", "expected": "KB-NET-003"},
    {"question": "I need permission to access a department folder.", "expected": "KB-NET-004"},
    {"question": "The mapped network drive has disappeared.", "expected": "KB-NET-005"},
    {"question": "Wireless internet works on other devices but not on mine.", "expected": "KB-WIFI-003"},
    {"question": "I need to recover access to my account after too many wrong attempts.", "expected": "KB-PASS-002"},
    {"question": "A monitor connected to my computer is blank and shows no signal.", "expected": "KB-HW-004"},
    {"question": "Can you suggest a recipe for chocolate cake?", "expected": None},
    {"question": "What is the current weather in Mumbai?", "expected": None},
    {"question": "Explain how to invest in the stock market.", "expected": None},
    {"question": "Write a poem about a dragon.", "expected": None},
]


def balanced_accuracy(labels: list[int], predictions: list[int]) -> float:
    """Average recall for answerable and out-of-scope classes."""
    positives = [i for i, label in enumerate(labels) if label == 1]
    negatives = [i for i, label in enumerate(labels) if label == 0]
    if not positives or not negatives:
        return 0.0
    sensitivity = sum(predictions[i] == 1 for i in positives) / len(positives)
    specificity = sum(predictions[i] == 0 for i in negatives) / len(negatives)
    return (sensitivity + specificity) / 2


def evaluate():
    """Build a fresh index, report Hit@3/MRR, and tune threshold from labelled queries."""
    build_index()
    rows = []
    for case in TEST_CASES:
        # Return up to ten chunks so MRR@10 can inspect the ranking beyond the top three.
        results = retrieve(case["question"], top_k=10)
        ids = [item["article_id"] for item in results]
        expected = case["expected"]
        if expected is None:
            reciprocal_rank = 0.0
            hit3 = 0
        else:
            rank = next((position for position, article_id in enumerate(ids, start=1)
                         if article_id == expected), None)
            reciprocal_rank = 1.0 / rank if rank is not None else 0.0
            hit3 = int(expected in ids[:3])
        rows.append({
            "question": case["question"], "expected": expected,
            "top_ids": ids, "best_similarity": results[0]["similarity"] if results else -1.0,
            "hit3": hit3, "reciprocal_rank": reciprocal_rank,
            "label": int(expected is not None),
        })

    in_scope = [row for row in rows if row["expected"] is not None]
    hit3_total = sum(row["hit3"] for row in in_scope)
    mrr = sum(row["reciprocal_rank"] for row in in_scope) / len(in_scope)
    print("\nRETRIEVAL METRICS (in-scope questions only)")
    print(f"In-scope questions: {len(in_scope)}")
    print(f"Hit@3: {hit3_total}/{len(in_scope)} = {hit3_total / len(in_scope):.1%}")
    print(f"MRR@10: {mrr:.4f}")

    # Tune a single best-match similarity threshold to classify answerable vs OOS.
    # Balanced accuracy prevents the 28/4 class imbalance from dominating the result.
    scores = sorted(set(row["best_similarity"] for row in rows))
    candidates = [scores[0] - 1e-6] + [
        (left + right) / 2 for left, right in zip(scores, scores[1:])
    ] + [scores[-1] + 1e-6]
    best_threshold, best_score = candidates[0], -1.0
    labels = [row["label"] for row in rows]
    for threshold in candidates:
        predictions = [int(row["best_similarity"] >= threshold) for row in rows]
        score = balanced_accuracy(labels, predictions)
        # In a tie, prefer the higher threshold to be conservative about unsupported answers.
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_threshold, best_score = threshold, score

    threshold_config = {
        "similarity_threshold": round(float(best_threshold), 6),
        "selection_method": "maximum balanced accuracy on this labelled demo set; ties prefer higher threshold",
        "balanced_accuracy": round(float(best_score), 4),
        "in_scope_queries": len(in_scope),
        "out_of_scope_queries": len(rows) - len(in_scope),
        "warning": "Threshold is tuned to this small demo set; validate on new unseen questions before real use."
    }
    THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(threshold_config, indent=2), encoding="utf-8")
    print("\nTHRESHOLD TUNING")
    print(f"Selected similarity threshold: {best_threshold:.4f}")
    print(f"Balanced accuracy for answerable/OOS detection: {best_score:.1%}")
    print(f"Saved threshold to: {THRESHOLD_FILE}")
    print("\nPER-QUESTION RESULTS")
    for row in rows:
        print(f"- {row['question']} | expected={row['expected']} | "
              f"top3={row['top_ids'][:3]} | best={row['best_similarity']:.3f} | "
              f"Hit@3={row['hit3']}")


if __name__ == "__main__":
    evaluate()
