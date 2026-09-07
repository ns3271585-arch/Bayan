"""Lab 2: Transformer anatomy and attention diagnostics."""

import math

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from bayan.attention import attention, MultiHeadAttention


def average_pad_mass(attentions, attention_mask):
    """Average attention mass assigned to PAD keys."""

    valid_queries = attention_mask.bool()
    pad_keys = ~valid_queries

    layer_values = []

    for attn in attentions:
        # attn: [batch, heads, query, key]
        pad_mass_per_query = (
            attn * pad_keys[:, None, None, :]
        ).sum(dim=-1)

        numerator = (
            pad_mass_per_query
            * valid_queries[:, None, :]
        ).sum()

        denominator = (
            valid_queries.sum() * attn.size(1)
        )

        layer_values.append(
            (numerator / denominator).item()
        )

    return sum(layer_values) / len(layer_values)


def find_adjacency_head(attentions, attention_mask):
    """Find the head paying the most attention to neighbouring tokens."""

    valid = attention_mask.bool()
    seq_len = attention_mask.size(1)

    adjacency = torch.zeros(
        seq_len,
        seq_len,
        dtype=torch.bool,
    )

    positions = torch.arange(seq_len - 1)

    adjacency[positions, positions + 1] = True
    adjacency[positions + 1, positions] = True

    best_score = -1.0
    best_layer = -1
    best_head = -1

    for layer_idx, attn in enumerate(attentions):

        valid_keys = valid[:, None, None, :]

        neighbour_mass = (
            attn
            * adjacency[None, None, :, :]
            * valid_keys
        ).sum(dim=-1)

        scores = (
            neighbour_mass
            * valid[:, None, :]
        ).sum(dim=(0, 2)) / valid.sum()

        head_idx = int(scores.argmax())
        score = float(scores[head_idx])

        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_head = head_idx

    return best_layer, best_head, best_score


def find_sep_sink(attentions, input_ids, attention_mask, sep_token_id):
    """Find the head assigning the most mass to SEP tokens."""

    valid = attention_mask.bool()
    sep_positions = input_ids.eq(sep_token_id)

    best_score = -1.0
    best_layer = -1
    best_head = -1

    for layer_idx, attn in enumerate(attentions):

        sep_mass = (
            attn
            * sep_positions[:, None, None, :]
        ).sum(dim=-1)

        scores = (
            sep_mass
            * valid[:, None, :]
        ).sum(dim=(0, 2)) / valid.sum()

        head_idx = int(scores.argmax())
        score = float(scores[head_idx])

        if score > best_score:
            best_score = score
            best_layer = layer_idx
            best_head = head_idx

    return best_layer, best_head, best_score


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("1. SCALED DOT-PRODUCT ATTENTION")
    print("=" * 70)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    our_output = attention(q, k, v)

    pytorch_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
    )

    max_diff = (
        our_output - pytorch_output
    ).abs().max().item()

    print(f"Output shape: {our_output.shape}")
    print(
        f"Maximum absolute difference vs PyTorch: "
        f"{max_diff:.10f}"
    )
    print(f"Equivalent within 1e-6: {max_diff < 1e-6}")

    print()
    print("=" * 70)
    print("2. ATTENTION WEIGHT MATRIX")
    print("=" * 70)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    ) / math.sqrt(q.size(-1))

    weights = torch.softmax(scores, dim=-1)

    print("Head 0 attention weights:")
    print(weights[0, 0])

    print()
    print("Row sums:")
    print(weights[0, 0].sum(dim=-1))

    print()
    print("=" * 70)
    print("3. MULTI-HEAD ATTENTION")
    print("=" * 70)

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
    )

    mha_output = mha(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {mha_output.shape}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {d_model // num_heads}")

    print()
    print("=" * 70)
    print("4. CAUSAL MASK")
    print("=" * 70)

    seq_len = 4

    causal_mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
        )
    )

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    masked_output = attention(
        q,
        k,
        v,
        mask=causal_mask,
    )

    masked_scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    ) / math.sqrt(q.size(-1))

    masked_scores = masked_scores.masked_fill(
        ~causal_mask,
        float("-inf"),
    )

    masked_weights = torch.softmax(
        masked_scores,
        dim=-1,
    )

    print("Causal mask:")
    print(causal_mask[0, 0].int())

    print()
    print("Masked attention weights - Head 0:")
    print(masked_weights[0, 0])

    upper_triangle = torch.triu(
        masked_weights[0, 0],
        diagonal=1,
    )

    print()
    print(
        "Future-token attention mass:",
        upper_triangle.sum().item(),
    )

    print(
        "Lower-triangular causal behaviour:",
        torch.allclose(
            upper_triangle,
            torch.zeros_like(upper_triangle),
        ),
    )

    print("Model family: Decoder-style causal attention")

    print()
    print("=" * 70)
    print("5. BAYAN ATTENTION-MAP + PAD LEAK DIAGNOSTICS")
    print("=" * 70)

    df = pd.read_csv(
        "data/raw/bayan_feedback.csv"
    )

    all_texts = (
        df["text"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # Use different-length Bayan examples so padding is present.
    all_texts = sorted(
        all_texts,
        key=lambda text: len(text.split()),
    )

    middle = len(all_texts) // 2

    bayan_examples = [
        all_texts[0],
        all_texts[1],
        all_texts[middle],
        all_texts[-2],
        all_texts[-1],
    ]

    checkpoint = "bert-base-multilingual-cased"

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = AutoModel.from_pretrained(
        checkpoint,
        attn_implementation="eager",
    )

    model.eval()

    encoded = tokenizer(
        bayan_examples,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():

        # Correct run: PAD tokens are masked.
        masked_result = model(
            **encoded,
            output_attentions=True,
            return_dict=True,
        )

        # Incorrect run: attention_mask intentionally omitted.
        no_mask_inputs = {
            key: value
            for key, value in encoded.items()
            if key != "attention_mask"
        }

        unmasked_result = model(
            **no_mask_inputs,
            output_attentions=True,
            return_dict=True,
        )

    masked_pad_mass = average_pad_mass(
        masked_result.attentions,
        encoded["attention_mask"],
    )

    unmasked_pad_mass = average_pad_mass(
        unmasked_result.attentions,
        encoded["attention_mask"],
    )

    adj_layer, adj_head, adj_score = find_adjacency_head(
        masked_result.attentions,
        encoded["attention_mask"],
    )

    sep_layer, sep_head, sep_score = find_sep_sink(
        masked_result.attentions,
        encoded["input_ids"],
        encoded["attention_mask"],
        tokenizer.sep_token_id,
    )

    total_pad_tokens = (
        encoded["attention_mask"] == 0
    ).sum().item()

    print(f"Bayan examples inspected: {len(bayan_examples)}")
    print(f"PAD tokens in diagnostic batch: {total_pad_tokens}")

    print()
    print(
        f"Average PAD attention mass WITH correct mask: "
        f"{masked_pad_mass:.8f}"
    )

    print(
        f"Average PAD attention mass WITHOUT mask: "
        f"{unmasked_pad_mass:.8f}"
    )

    print()
    print(
        "Best adjacency-looking head: "
        f"layer {adj_layer}, head {adj_head}, "
        f"adjacency mass = {adj_score:.6f}"
    )

    print(
        "Strongest [SEP] sink head: "
        f"layer {sep_layer}, head {sep_head}, "
        f"SEP mass = {sep_score:.6f}"
    )

    print()

    if unmasked_pad_mass > masked_pad_mass:
        print(
            "PAD leak detected when attention mask is omitted."
        )
    else:
        print(
            "No increased PAD mass detected in this diagnostic batch."
        )


if __name__ == "__main__":
    main()