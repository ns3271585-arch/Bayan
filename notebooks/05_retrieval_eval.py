"""Lab 5: labelled-query retrieval evaluation."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bayan.search.service import CaseSearch


PREFIX = ROOT / "artifacts" / "search" / "case_index_v1"
QUERIES_PATH = ROOT / "data" / "search" / "bayan_queries.jsonl"
CASES_PATH = ROOT / "data" / "search" / "bayan_cases.csv"
RESULTS_PATH = ROOT / "artifacts" / "search" / "retrieval_eval.json"

TOP_K = 10
CANDIDATES = 50
NO_ANSWER_THRESHOLD = 0.25


def load_queries():
    rows = []

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def load_case_languages():
    cases = pd.read_csv(
        CASES_PATH,
        usecols=["case_id", "lang"],
    )

    return dict(
        zip(
            cases["case_id"],
            cases["lang"],
        )
    )


def recall_at_10(retrieved_ids, relevant_ids):
    """
    Course-compatible query-level recall@10.

    Returns 1 if at least one judged relevant case occurs in the top 10,
    otherwise 0.
    """
    gold = set(relevant_ids)

    return float(
        any(
            case_id in gold
            for case_id in retrieved_ids[:TOP_K]
        )
    )


def strict_recall_at_10(retrieved_ids, relevant_ids):
    """Diagnostic: fraction of judged relevant IDs recovered in the top 10."""
    gold = set(relevant_ids)

    if not gold:
        return 0.0

    hits = sum(
        case_id in gold
        for case_id in retrieved_ids[:TOP_K]
    )

    return hits / len(gold)


def mrr_at_10(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first judged relevant case in the top 10."""
    gold = set(relevant_ids)

    for rank, case_id in enumerate(
        retrieved_ids[:TOP_K],
        start=1,
    ):
        if case_id in gold:
            return 1.0 / rank

    return 0.0


def mean(values):
    return float(np.mean(values)) if values else 0.0


def p50_ms(values):
    return float(np.median(values)) if values else 0.0


def evaluate_answerable(
    searcher,
    queries,
    case_languages,
):
    rows = []
    bi_latencies = []
    full_latencies = []

    for number, q in enumerate(queries, start=1):
        bi_start = time.perf_counter()

        _, bi_results = searcher._retrieve(
            q["query"],
            candidates=TOP_K,
        )

        bi_end = time.perf_counter()

        bi_ids = [
            item["case_id"]
            for item in bi_results
        ]

        full_start = time.perf_counter()

        normalized_query, candidates = searcher._retrieve(
            q["query"],
            candidates=CANDIDATES,
        )

        pairs = [
            (normalized_query, item["indexed_text"])
            for item in candidates
        ]

        ce_scores = searcher.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked = []

        for item, score in zip(candidates, ce_scores):
            copy = dict(item)
            copy["score"] = float(score)
            reranked.append(copy)

        reranked.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        after_rerank = time.perf_counter()

        reranked_ids = [
            item["case_id"]
            for item in reranked[:TOP_K]
        ]

        same_language_gold = [
            case_id
            for case_id in q["relevant_case_ids"]
            if case_languages[case_id] == q["lang"]
        ]

        cross_language_gold = [
            case_id
            for case_id in q["relevant_case_ids"]
            if case_languages[case_id] != q["lang"]
        ]

        bi_latency = (bi_end - bi_start) * 1000.0
        full_latency = (after_rerank - full_start) * 1000.0

        bi_latencies.append(bi_latency)
        full_latencies.append(full_latency)

        rows.append(
            {
                "query_id": q["query_id"],
                "lang": q["lang"],
                "bi_recall_at_10": recall_at_10(
                    bi_ids,
                    q["relevant_case_ids"],
                ),
                "bi_strict_recall_at_10": strict_recall_at_10(
                    bi_ids,
                    q["relevant_case_ids"],
                ),
                "bi_mrr_at_10": mrr_at_10(
                    bi_ids,
                    q["relevant_case_ids"],
                ),
                "rerank_recall_at_10": recall_at_10(
                    reranked_ids,
                    q["relevant_case_ids"],
                ),
                "rerank_strict_recall_at_10": strict_recall_at_10(
                    reranked_ids,
                    q["relevant_case_ids"],
                ),
                "rerank_mrr_at_10": mrr_at_10(
                    reranked_ids,
                    q["relevant_case_ids"],
                ),
                "bi_same_recall_at_10": recall_at_10(
                    bi_ids,
                    same_language_gold,
                ),
                "bi_same_mrr_at_10": mrr_at_10(
                    bi_ids,
                    same_language_gold,
                ),
                "bi_cross_recall_at_10": recall_at_10(
                    bi_ids,
                    cross_language_gold,
                ),
                "bi_cross_mrr_at_10": mrr_at_10(
                    bi_ids,
                    cross_language_gold,
                ),
                "rerank_same_recall_at_10": recall_at_10(
                    reranked_ids,
                    same_language_gold,
                ),
                "rerank_same_mrr_at_10": mrr_at_10(
                    reranked_ids,
                    same_language_gold,
                ),
                "rerank_cross_recall_at_10": recall_at_10(
                    reranked_ids,
                    cross_language_gold,
                ),
                "rerank_cross_mrr_at_10": mrr_at_10(
                    reranked_ids,
                    cross_language_gold,
                ),
                "bi_latency_ms": bi_latency,
                "full_latency_ms": full_latency,
            }
        )

        if number % 10 == 0 or number == len(queries):
            print(
                f"Evaluated {number}/{len(queries)} answerable queries"
            )

    return rows, bi_latencies, full_latencies


def summarize_slice(rows, lang=None):
    selected = rows

    if lang is not None:
        selected = [
            row
            for row in rows
            if row["lang"] == lang
        ]

    return {
        "n_queries": len(selected),
        "bi_recall_at_10": mean(
            [row["bi_recall_at_10"] for row in selected]
        ),
        "bi_strict_recall_at_10": mean(
            [row["bi_strict_recall_at_10"] for row in selected]
        ),
        "bi_mrr_at_10": mean(
            [row["bi_mrr_at_10"] for row in selected]
        ),
        "rerank_recall_at_10": mean(
            [row["rerank_recall_at_10"] for row in selected]
        ),
        "rerank_strict_recall_at_10": mean(
            [row["rerank_strict_recall_at_10"] for row in selected]
        ),
        "rerank_mrr_at_10": mean(
            [row["rerank_mrr_at_10"] for row in selected]
        ),
    }


def summarize_language_alignment(rows):
    return {
        "bi_same_recall_at_10": mean(
            [row["bi_same_recall_at_10"] for row in rows]
        ),
        "bi_same_mrr_at_10": mean(
            [row["bi_same_mrr_at_10"] for row in rows]
        ),
        "bi_cross_recall_at_10": mean(
            [row["bi_cross_recall_at_10"] for row in rows]
        ),
        "bi_cross_mrr_at_10": mean(
            [row["bi_cross_mrr_at_10"] for row in rows]
        ),
        "rerank_same_recall_at_10": mean(
            [row["rerank_same_recall_at_10"] for row in rows]
        ),
        "rerank_same_mrr_at_10": mean(
            [row["rerank_same_mrr_at_10"] for row in rows]
        ),
        "rerank_cross_recall_at_10": mean(
            [row["rerank_cross_recall_at_10"] for row in rows]
        ),
        "rerank_cross_mrr_at_10": mean(
            [row["rerank_cross_mrr_at_10"] for row in rows]
        ),
    }


def evaluate_no_answer(searcher, queries):
    correct = 0
    latencies = []

    for q in queries:
        start = time.perf_counter()

        results = searcher.search(
            q["query"],
            k=TOP_K,
            candidates=CANDIDATES,
            min_score=NO_ANSWER_THRESHOLD,
            rerank=True,
        )

        elapsed = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed)

        if len(results) == 0:
            correct += 1

    return correct, latencies


def main():
    queries = load_queries()
    case_languages = load_case_languages()

    answerable = [
        q
        for q in queries
        if not q["no_answer"]
    ]

    no_answer = [
        q
        for q in queries
        if q["no_answer"]
    ]

    print("Loading search service...")
    searcher = CaseSearch(str(PREFIX))

    print()
    print("Evaluating answerable queries...")

    rows, bi_latencies, full_latencies = evaluate_answerable(
        searcher,
        answerable,
        case_languages,
    )

    overall = summarize_slice(rows)
    arabic = summarize_slice(rows, "ar")
    english = summarize_slice(rows, "en")
    alignment = summarize_language_alignment(rows)

    recall_gap = (
        alignment["rerank_same_recall_at_10"]
        - alignment["rerank_cross_recall_at_10"]
    )

    mrr_gap = (
        alignment["rerank_same_mrr_at_10"]
        - alignment["rerank_cross_mrr_at_10"]
    )

    print()
    print("Evaluating no-answer threshold...")

    no_answer_correct, no_answer_latencies = evaluate_no_answer(
        searcher,
        no_answer,
    )

    summary = {
        "configuration": {
            "top_k": TOP_K,
            "candidates": CANDIDATES,
            "min_score": NO_ANSWER_THRESHOLD,
            "encoder": searcher.manifest["model"],
            "reranker": searcher.reranker_name,
            "preproc_version": searcher.manifest["preproc_version"],
        },
        "overall": overall,
        "arabic_query_slice": arabic,
        "english_query_slice": english,
        "language_alignment": alignment,
        "cross_lingual_gap": {
            "definition": (
                "same-language minus cross-language judged relevance"
            ),
            "rerank_recall_gap": recall_gap,
            "rerank_mrr_gap": mrr_gap,
        },
        "latency_ms": {
            "bi_encoder_p50": p50_ms(bi_latencies),
            "two_stage_p50": p50_ms(full_latencies),
            "no_answer_p50": p50_ms(no_answer_latencies),
        },
        "no_answer": {
            "correct": no_answer_correct,
            "total": len(no_answer),
            "threshold": NO_ANSWER_THRESHOLD,
        },
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 66)
    print("LAB 5 RETRIEVAL EVALUATION")
    print("=" * 66)

    print()
    print("OVERALL")
    print(
        f"Bi-encoder recall@10: "
        f"{overall['bi_recall_at_10']:.4f}"
    )
    print(
        f"Bi-encoder MRR@10:    "
        f"{overall['bi_mrr_at_10']:.4f}"
    )
    print(
        f"Reranked recall@10:   "
        f"{overall['rerank_recall_at_10']:.4f}"
    )
    print(
        f"Reranked MRR@10:      "
        f"{overall['rerank_mrr_at_10']:.4f}"
    )

    print()
    print("STRICT RECALL DIAGNOSTIC")
    print(
        f"Bi-encoder strict recall@10: "
        f"{overall['bi_strict_recall_at_10']:.4f}"
    )
    print(
        f"Reranked strict recall@10:   "
        f"{overall['rerank_strict_recall_at_10']:.4f}"
    )

    print()
    print("QUERY-LANGUAGE SLICES")
    print(
        "Arabic  rerank recall/MRR: "
        f"{arabic['rerank_recall_at_10']:.4f} / "
        f"{arabic['rerank_mrr_at_10']:.4f}"
    )
    print(
        "English rerank recall/MRR: "
        f"{english['rerank_recall_at_10']:.4f} / "
        f"{english['rerank_mrr_at_10']:.4f}"
    )

    print()
    print("SAME-LANGUAGE VS CROSS-LANGUAGE GOLD")
    print(
        "Same-language rerank recall/MRR:  "
        f"{alignment['rerank_same_recall_at_10']:.4f} / "
        f"{alignment['rerank_same_mrr_at_10']:.4f}"
    )
    print(
        "Cross-language rerank recall/MRR: "
        f"{alignment['rerank_cross_recall_at_10']:.4f} / "
        f"{alignment['rerank_cross_mrr_at_10']:.4f}"
    )
    print(
        f"Cross-lingual recall gap: {recall_gap:.4f}"
    )
    print(
        f"Cross-lingual MRR gap:    {mrr_gap:.4f}"
    )

    print()
    print("NO-ANSWER")
    print(
        f"Empty-correct: "
        f"{no_answer_correct}/{len(no_answer)} "
        f"at min_score={NO_ANSWER_THRESHOLD}"
    )

    print()
    print("P50 LATENCY")
    print(
        f"Bi-encoder only: "
        f"{p50_ms(bi_latencies):.2f} ms/query"
    )
    print(
        f"Two-stage:       "
        f"{p50_ms(full_latencies):.2f} ms/query"
    )

    print()
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
