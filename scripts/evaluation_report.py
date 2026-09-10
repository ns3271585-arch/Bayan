"""Lab 6: generate the evaluation report and model-card evidence."""

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from bayan.evaluation.bootstrap import bootstrap_ci
from bayan.evaluation.slices import sliced_report


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = ROOT / "data/eval/validation_predictions.csv"
MANUAL_ERRORS = ROOT / "manual_error_sample.csv"
TEMPLATE_DIR = ROOT / "templates"
MODEL_CARD_DIR = ROOT / "docs/model_cards"
REPORT_PATH = ROOT / "EVALUATION_REPORT.md"


BEHAVIOURAL = {
    "XLM-R incumbent": {
        "invariance": (140, 200),
        "mft": (15, 16),
        "directional": None,
    },
    "CAMeLBERT-DA": {
        "invariance": (140, 200),
        "mft": (5, 16),
        "directional": None,
    },
    "CAMeLBERT-mix": {
        "invariance": None,
        "mft": None,
        "directional": None,
    },
}


MODELS = [
    {
        "model_name": "XLM-R incumbent",
        "checkpoint": "xlm-roberta-base",
        "intended_use": (
            "Bilingual Arabic/English feedback topic classification "
            "for the Bayan service-feedback domain."
        ),
        "metrics": {
            "All Arabic macro-F1": "1.0000 [1.0000, 1.0000]",
            "Gulf macro-F1": "1.0000 [1.0000, 1.0000]",
            "MSA macro-F1": "1.0000",
            "Arabic tokenizer fertility": "1.685",
        },
        "limitations": [
            "The behavioural invariance pass rate is only 70.0%, below the 95% Lab 6 target.",
            "The Arabic-only MFT slice passes 7/8 cases, so basic Arabic topic behaviour is not perfect.",
            "The supplied directional tests require sentiment scoring, but no trained sentiment scorer is available in the current project artefacts.",
            "Perfect frozen-test macro-F1 should not be interpreted as proof of robustness outside the supplied synthetic/course dataset.",
        ],
    },
    {
        "model_name": "CAMeLBERT-mix",
        "checkpoint": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        "intended_use": (
            "Arabic feedback topic classification, used as an "
            "Arabic-centric comparison model in the Lab 4 bake-off."
        ),
        "metrics": {
            "All Arabic macro-F1": "1.0000",
            "Gulf macro-F1": "1.0000",
            "MSA macro-F1": "1.0000",
            "Arabic tokenizer fertility": "1.412",
        },
        "limitations": [
            "This Arabic-centric model is not intended as the primary English classifier.",
            "A separate Lab 6 behavioural run was not recorded for this checkpoint, so behavioural rates are reported as not measured.",
            "Perfect frozen-test metrics may reflect the constrained course dataset and do not establish robustness on unseen real-world distributions.",
            "Production use would require fresh evaluation under serving, latency, and distribution-shift conditions.",
        ],
    },
    {
        "model_name": "CAMeLBERT-DA",
        "checkpoint": "CAMeL-Lab/bert-base-arabic-camelbert-da",
        "intended_use": (
            "Arabic and dialect-aware feedback topic classification, "
            "with emphasis on Gulf and Arabic evaluation slices."
        ),
        "metrics": {
            "All Arabic macro-F1": "1.0000 [1.0000, 1.0000]",
            "Gulf macro-F1": "1.0000 [1.0000, 1.0000]",
            "MSA macro-F1": "1.0000",
            "Arabic tokenizer fertility": "1.412",
        },
        "limitations": [
            "The invariance pass rate is 70.0%, below the 95% Lab 6 target.",
            "The full student-authored MFT suite passes only 5/16 cases (31.25%).",
            "English MFT performance is especially weak because this checkpoint was trained on the Arabic slice.",
            "The supplied directional tests cannot be scored without a trained sentiment scorer.",
            "Perfect frozen-test macro-F1 does not remove the behavioural robustness limitations observed in Lab 6.",
        ],
    },
]


def md_table(rows):
    if not rows:
        return "_No results available._"

    frame = pd.DataFrame(rows)
    return frame.to_markdown(index=False)


def behavioural_table(model_name):
    values = BEHAVIOURAL[model_name]
    rows = []

    for test_name in ("invariance", "mft", "directional"):
        result = values[test_name]

        if result is None:
            rate = "N/A"
        else:
            passed, total = result
            rate = f"{passed}/{total} ({100 * passed / total:.2f}%)"

        rows.append(
            {
                "Test": test_name,
                "Result": rate,
            }
        )

    return md_table(rows)


def metrics_table(model):
    return md_table(
        [
            {"Metric": name, "Result": value}
            for name, value in model["metrics"].items()
        ]
    )
def load_manual_limitations(card_path):
    """Preserve the hand-written Known limitations section on regeneration."""
    if not card_path.exists():
        return "TODO — write this section by hand."

    text = card_path.read_text(encoding="utf-8")

    start_marker = "## Known limitations"
    end_marker = "## Contact / owner"

    if start_marker not in text or end_marker not in text:
        return "TODO — write this section by hand."

    section = (
        text.split(start_marker, 1)[1]
        .split(end_marker, 1)[0]
        .strip()
    )

    return section or "TODO — write this section by hand."

def generate_model_cards(slice_table):
    MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("model_card.md.j2")

    generated = []

    for model in MODELS:
        safe_name = (
            model["model_name"]
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        output = MODEL_CARD_DIR / f"{safe_name}.md"
        if model["model_name"] == "XLM-R incumbent":
            model_slice_table = slice_table
        else:
            model_slice_table = (
                "_No model-specific Lab 6 validation slice predictions were "
                "recorded for this checkpoint. Arabic/Gulf/MSA frozen-test "
                "metrics are reported in the Metrics section above._"
            )
               
        card = template.render(
            model_name=model["model_name"],
            intended_use=model["intended_use"],
            checkpoint=model["checkpoint"],
            preproc_version="1.2.0",
            data_version="Bayan course dataset / frozen Lab 3 split",
            metrics_table=metrics_table(model),
            slices_table=model_slice_table,
            behavioural_table=behavioural_table(model["model_name"]),
            known_limitations=load_manual_limitations(output),
            owner="Bayan course team",
        )

        output.write_text(card, encoding="utf-8")
        generated.append(output)

    return generated


def main():
    predictions = pd.read_csv(PREDICTIONS)

    correct = (
        predictions["y_true"].astype(str)
        == predictions["y_pred"].astype(str)
    ).astype(float)

    point, ci_low, ci_high = bootstrap_ci(
        correct.to_numpy(),
        n_boot=2000,
        seed=42,
    )

    slices = sliced_report(
        predictions,
        n_boot=1000,
        seed=42,
    )

    slice_rows = slices[
        [
            "slice_type",
            "slice_value",
            "n",
            "accuracy",
            "ci_low",
            "ci_high",
            "small_slice",
        ]
    ].copy()

    for column in ["accuracy", "ci_low", "ci_high"]:
        slice_rows[column] = slice_rows[column].map(
            lambda x: f"{x:.4f}"
        )

    slice_table = slice_rows.to_markdown(index=False)

    errors = pd.read_csv(MANUAL_ERRORS)

    if len(errors) != 120:
        raise ValueError(
            f"Expected 120 manually reviewed errors, found {len(errors)}"
        )

    if errors["error_category"].isna().any():
        raise ValueError("All 120 errors must be manually tagged.")

    taxonomy = (
        errors["error_category"]
        .value_counts()
        .rename_axis("Error category")
        .reset_index(name="Count")
    )
    taxonomy["Share"] = (
        taxonomy["Count"] / len(errors) * 100
    ).map(lambda x: f"{x:.1f}%")

    taxonomy_table = taxonomy.to_markdown(index=False)

    cards = generate_model_cards(slice_table)

    report = f"""# Bayan Lab 6 Evaluation Report

## Executive Summary

Validation accuracy is {point:.1%} with a 95% bootstrap confidence interval
of [{ci_low:.1%}, {ci_high:.1%}].

The sliced evaluation exposes a substantial language and class-level gap:
Arabic accuracy is lower than English accuracy, and the `parks` class is the
dominant observed topic-classification failure.

Behavioural evaluation also shows that perfect frozen-test macro-F1 does not
imply robust behaviour. Both evaluated models achieve only 70.0% invariance.

## Bootstrap Result

| Metric | Point estimate | 95% CI |
|---|---:|---:|
| Validation accuracy | {point:.4f} | [{ci_low:.4f}, {ci_high:.4f}] |

## Sliced Evaluation

{slice_table}

## Model Comparison

| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---:|---:|---:|---:|
| XLM-R incumbent | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 93.75% |
| CAMeLBERT-DA | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 70.00% | 31.25% |

The paired XLM-R minus CAMeLBERT-DA macro-F1 difference is 0.0000 with
bootstrap interval [0.0000, 0.0000] on both the full Arabic and Gulf frozen
test slices. The quality tie therefore does not justify replacing the XLM-R
incumbent on aggregate quality alone.

## Behavioural Evaluation

### XLM-R incumbent

{behavioural_table("XLM-R incumbent")}

### CAMeLBERT-DA

{behavioural_table("CAMeLBERT-DA")}

The Lab 6 invariance target is 95%, so both evaluated models miss the target.

The MFT target is 90%. XLM-R passes the full 16-case student-authored MFT
suite at 93.75%, while CAMeLBERT-DA passes 31.25%. The Arabic-only MFT rates
are 87.5% for XLM-R and 50.0% for CAMeLBERT-DA.

The supplied directional templates describe a sentiment relation, but the
current project artefacts do not contain a trained sentiment scoring model.
Directional cases are therefore reported as not evaluated rather than being
assigned an artificial pass rate.

The 16 MFT cases were authored for Lab 6 from the service names in the
project service registry because the supplied behavioural template file
contains invariance and directional cases but no minimum-functionality rows.

## Manual Error Taxonomy

A reproducible random sample of 120 errors was drawn from the 300 validation
errors with `random_state=42`. All 120 sampled errors were read and tagged.

{taxonomy_table}

The dominant pattern is systematic `parks -> roads` confusion. The next
largest category is ambiguity around park walkways, followed by Arabic
orthographic variation.

## Top 3 Prioritised Fixes

1. **Target `parks -> roads` confusion.** Add hard-negative and contrastive
   park-versus-road training examples. Estimated upper-bound validation
   accuracy delta: **+7.0 percentage points**.

2. **Improve contextual disambiguation of park walkways.** Strengthen examples
   in which `الممر` occurs with clear park context such as `حديقة`.
   Estimated upper-bound validation accuracy delta:
   **+4.2 percentage points**.

3. **Strengthen Arabic orthographic normalisation and augmentation.** Cover
   elongation and observed alef/hamza spelling variation.
   Estimated upper-bound validation accuracy delta:
   **+1.4 percentage points**.

These deltas are scenario estimates derived from the manually reviewed error
shares. They are not measured improvements and assume that the corresponding
error category can be corrected without introducing regressions.

## Model Cards

Three model cards were generated:

- `docs/model_cards/xlm_r_incumbent.md`
- `docs/model_cards/camelbert_mix.md`
- `docs/model_cards/camelbert_da.md`

## Known Evaluation Limitations

- The validation prediction file contains a concentrated `parks -> roads`
  failure, so aggregate accuracy hides a severe class-level weakness.
- Behavioural invariance is below target for both evaluated models.
- Directional sentiment behaviour cannot be measured from the currently
  trained model artefacts.
- The MFT cases are student-authored because minimum-functionality rows were
  not supplied in the behavioural template CSV.
- Frozen-test scores of 1.0000 should not be treated as evidence of
  production robustness.
- Predicted metric deltas for proposed fixes are estimates and require
  retraining to verify.

## Evidence Summary

- Bootstrap evaluation: complete.
- Sliced evaluation: complete.
- Behavioural evaluation: invariance and MFT measured; directional documented
  as unavailable with current model artefacts.
- Manual error review: 120/120 tagged.
- Error taxonomy and top fixes: complete.
- Model cards: 3 generated.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Wrote: {REPORT_PATH}")
    for card in cards:
        print(f"Wrote: {card}")


if __name__ == "__main__":
    main()