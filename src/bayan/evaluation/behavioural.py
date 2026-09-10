"""Lab 6: behavioural evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


REQUIRED_TEMPLATE_COLUMNS = {
    "test_id",
    "test_type",
    "lang",
    "template",
    "term",
    "expected_relation",
}


def _load_frame(data):
    """Accept either a DataFrame or a CSV path."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    return pd.read_csv(Path(data))


def _render(template, term):
    """Insert the supplied perturbation term into a behavioural template."""
    return str(template).replace("{term}", str(term))


def _positive_control(text, lang):
    """
    Construct the non-negated control for the supplied directional skeletons.
    """
    if lang == "en":
        marker = " is not working"
        if marker in text:
            return text.replace(marker, " is working", 1)

    if lang == "ar":
        marker = " لا تعمل"
        if marker in text:
            return text.replace(marker, " تعمل", 1)

    return None


def _summarize(details):
    """Summarise evaluated behavioural cases by test type."""
    rows = []

    for test_type in ["invariance", "directional", "mft"]:
        subset = details[
            details["test_type"].eq(test_type)
        ].copy()

        evaluated = subset[
            subset["evaluated"].eq(True)
        ]

        n_available = int(len(subset))
        n_evaluated = int(len(evaluated))
        n_passed = int(evaluated["passed"].sum()) if n_evaluated else 0

        pass_rate = (
            n_passed / n_evaluated
            if n_evaluated
            else None
        )

        rows.append(
            {
                "test_type": test_type,
                "available": n_available,
                "evaluated": n_evaluated,
                "passed": n_passed,
                "pass_rate": pass_rate,
            }
        )

    return pd.DataFrame(rows)


def run_behavioural_suite(
    templates,
    *,
    topic_predictor: Callable | None = None,
    sentiment_score_fn: Callable | None = None,
    mft_cases=None,
):
    """
    Run Bayan behavioural checks.

    Parameters
    ----------
    templates:
        Behavioural template DataFrame or CSV path.

    topic_predictor:
        Callable accepting a list of texts and returning one topic label
        per text.

    sentiment_score_fn:
        Optional callable accepting a list of texts and returning one
        numeric sentiment score per text. Larger values must mean more
        positive sentiment.

    mft_cases:
        Optional DataFrame or CSV path containing at least:
        ``text`` and ``expected_topic``.

    Returns
    -------
    dict
        ``details``: case-level results.
        ``summary``: pass/failure rates by behavioural test type.

    Notes
    -----
    The supplied course behavioural CSV contains invariance and
    directional skeletons, but no MFT cases. Missing predictors or
    missing MFT cases are therefore reported as not evaluated rather
    than assigned fabricated pass/failure scores.
    """
    df = _load_frame(templates)

    missing = REQUIRED_TEMPLATE_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "templates is missing required columns: "
            + ", ".join(sorted(missing))
        )

    detail_rows = []

    # ------------------------------------------------------------
    # Invariance: changing the inserted term should not change topic.
    # ------------------------------------------------------------

    invariance = df[
        df["test_type"].str.lower().eq("invariance")
    ].copy()

    if not invariance.empty:
        texts = [
            _render(row.template, row.term)
            for row in invariance.itertuples()
        ]

        predictions = None

        if topic_predictor is not None:
            predictions = list(topic_predictor(texts))

            if len(predictions) != len(invariance):
                raise ValueError(
                    "topic_predictor returned the wrong number of predictions"
                )

        invariance["rendered_text"] = texts

        if predictions is not None:
            invariance["prediction"] = predictions

            group_columns = [
                "lang",
                "template",
                "expected_relation",
            ]

            invariance["reference_prediction"] = (
                invariance.groupby(
                    group_columns,
                    sort=False,
                )["prediction"]
                .transform("first")
            )

            invariance["passed"] = (
                invariance["prediction"]
                == invariance["reference_prediction"]
            )

            invariance["evaluated"] = True
        else:
            invariance["prediction"] = None
            invariance["reference_prediction"] = None
            invariance["passed"] = False
            invariance["evaluated"] = False

        for row in invariance.itertuples():
            detail_rows.append(
                {
                    "test_id": row.test_id,
                    "test_type": "invariance",
                    "lang": row.lang,
                    "text": row.rendered_text,
                    "expected_relation": row.expected_relation,
                    "prediction": row.prediction,
                    "reference_prediction": row.reference_prediction,
                    "passed": bool(row.passed),
                    "evaluated": bool(row.evaluated),
                }
            )

    # ------------------------------------------------------------
    # Directional: negation must not improve sentiment.
    # ------------------------------------------------------------

    directional = df[
        df["test_type"].str.lower().eq("directional")
    ].copy()

    if not directional.empty:
        negative_texts = [
            _render(row.template, row.term)
            for row in directional.itertuples()
        ]

        positive_texts = [
            _positive_control(text, lang)
            for text, lang in zip(
                negative_texts,
                directional["lang"],
            )
        ]

        if sentiment_score_fn is not None:
            if any(text is None for text in positive_texts):
                raise ValueError(
                    "Could not construct a positive control for "
                    "one or more directional templates"
                )

            positive_scores = list(
                sentiment_score_fn(positive_texts)
            )
            negative_scores = list(
                sentiment_score_fn(negative_texts)
            )

            if (
                len(positive_scores) != len(directional)
                or len(negative_scores) != len(directional)
            ):
                raise ValueError(
                    "sentiment_score_fn returned the wrong "
                    "number of scores"
                )

            for i, row in enumerate(directional.itertuples()):
                passed = (
                    float(negative_scores[i])
                    <= float(positive_scores[i])
                )

                detail_rows.append(
                    {
                        "test_id": row.test_id,
                        "test_type": "directional",
                        "lang": row.lang,
                        "text": negative_texts[i],
                        "expected_relation": row.expected_relation,
                        "positive_score": float(positive_scores[i]),
                        "negative_score": float(negative_scores[i]),
                        "passed": bool(passed),
                        "evaluated": True,
                    }
                )
        else:
            for i, row in enumerate(directional.itertuples()):
                detail_rows.append(
                    {
                        "test_id": row.test_id,
                        "test_type": "directional",
                        "lang": row.lang,
                        "text": negative_texts[i],
                        "expected_relation": row.expected_relation,
                        "positive_score": None,
                        "negative_score": None,
                        "passed": False,
                        "evaluated": False,
                    }
                )

    # ------------------------------------------------------------
    # Minimum-functionality tests.
    # ------------------------------------------------------------

    if mft_cases is not None:
        mft = _load_frame(mft_cases)

        required_mft = {
            "text",
            "expected_topic",
        }

        missing_mft = required_mft - set(mft.columns)

        if missing_mft:
            raise ValueError(
                "mft_cases is missing required columns: "
                + ", ".join(sorted(missing_mft))
            )

        if topic_predictor is None:
            predictions = [None] * len(mft)
        else:
            predictions = list(
                topic_predictor(
                    mft["text"].astype(str).tolist()
                )
            )

            if len(predictions) != len(mft):
                raise ValueError(
                    "topic_predictor returned the wrong "
                    "number of MFT predictions"
                )

        for i, row in enumerate(mft.itertuples()):
            evaluated = topic_predictor is not None
            passed = (
                predictions[i] == row.expected_topic
                if evaluated
                else False
            )

            detail_rows.append(
                {
                    "test_id": (
                        getattr(row, "test_id", None)
                        or f"MFT-{i + 1:03d}"
                    ),
                    "test_type": "mft",
                    "lang": getattr(row, "lang", None),
                    "text": str(row.text),
                    "expected_relation": (
                        f"topic == {row.expected_topic}"
                    ),
                    "prediction": predictions[i],
                    "expected_topic": row.expected_topic,
                    "passed": bool(passed),
                    "evaluated": bool(evaluated),
                }
            )

    details = pd.DataFrame(detail_rows)

    if details.empty:
        details = pd.DataFrame(
            columns=[
                "test_id",
                "test_type",
                "lang",
                "text",
                "expected_relation",
                "passed",
                "evaluated",
            ]
        )

    summary = _summarize(details)

    return {
        "details": details,
        "summary": summary,
    }