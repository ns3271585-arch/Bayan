"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    """Bucket model parameters by major Transformer component."""

    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters():
        count = parameter.numel()

        lower_name = name.lower()

        # Layer normalization parameters
        if "layernorm" in lower_name or "layer_norm" in lower_name:
            buckets["norms"] += count

        # Embedding tables
        elif "embeddings" in lower_name:
            buckets["embeddings"] += count

        # Self-attention projections and attention output
        elif "attention" in lower_name:
            buckets["attention"] += count

        # Feed-forward network
        elif "intermediate" in lower_name or "output.dense" in lower_name:
            buckets["ffn"] += count

        # Pooler
        elif "pooler" in lower_name:
            buckets["pooler"] += count

        # Anything not captured above
        else:
            buckets["other"] += count

    total = sum(buckets.values())

    result = {
        "total_params": total,
    }

    for bucket, count in buckets.items():
        result[bucket] = count
        result[f"{bucket}_pct"] = (
            100.0 * count / total
            if total > 0
            else 0.0
        )

    return result


if __name__ == "__main__":
    checkpoints = [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]

    for ckpt in checkpoints:
        print("=" * 80)
        print(ckpt)
        print("=" * 80)

        result = audit(ckpt)

        print(f"Total parameters: {result['total_params']:,}")

        for bucket in [
            "embeddings",
            "attention",
            "ffn",
            "norms",
            "pooler",
            "other",
        ]:
            print(
                f"{bucket:12s}: "
                f"{result[bucket]:,} "
                f"({result[bucket + '_pct']:.2f}%)"
            )

        print()