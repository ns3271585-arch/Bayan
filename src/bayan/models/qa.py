"""Lab 3B: QA span selection with honest null handling."""

import numpy as np


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    """Return the best valid answer span or an explicit null answer."""

    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    start_indices = np.argsort(start_logits)[-top_k:][::-1]
    end_indices = np.argsort(end_logits)[-top_k:][::-1]

    best_score = float("-inf")
    best_start = None
    best_end = None
    best_start_index = None
    best_end_index = None

    for start_index in start_indices:
        for end_index in end_indices:

            # Ignore special tokens
            if offsets[start_index] is None:
                continue

            if offsets[end_index] is None:
                continue

            # Reject inverted spans
            if end_index < start_index:
                continue

            # Reject spans that are too long
            if end_index - start_index + 1 > max_answer_len:
                continue

            start_char = offsets[start_index][0]
            end_char = offsets[end_index][1]

            if end_char < start_char:
                continue

            score = (
                float(start_logits[start_index])
                + float(end_logits[end_index])
            )

            if score > best_score:
                best_score = score
                best_start = start_char
                best_end = end_char
                best_start_index = int(start_index)
                best_end_index = int(end_index)

    # No valid answer span
    if best_start is None:
        return {
            "answer": None,
            "start": None,
            "end": None,
            "score": None,
            "start_index": None,
            "end_index": None,
        }

    # Honest null prediction
    if null_score - best_score > null_threshold:
        return {
            "answer": None,
            "start": None,
            "end": None,
            "score": float(null_score),
            "start_index": None,
            "end_index": None,
        }

    # Valid answer span
    return {
        "answer": (best_start, best_end),
        "start": best_start,
        "end": best_end,
        "score": best_score,
        "start_index": best_start_index,
        "end_index": best_end_index,
    }