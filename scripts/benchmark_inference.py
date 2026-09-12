
"""Lab 7: honest CPU p50/p99 benchmark harness over production length mix.

Measures model inference only:
- model loading is excluded
- tokenization is excluded from the timed region
- warm-up requests are excluded
- CPU thread count is pinned
"""

import argparse
import gc
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_BENCH = "data/serving/bench_mix.npy"
DEFAULT_THREADS = 4
DEFAULT_WARMUP = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayan Lab 7 CPU inference benchmark."
    )

    parser.add_argument(
        "--pytorch-model",
        required=True,
        help="Path to the trained Hugging Face classifier checkpoint.",
    )

    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Tokenizer directory. Defaults to --pytorch-model.",
    )

    parser.add_argument(
        "--onnx-fp32",
        required=True,
        help="Path to the FP32 ONNX model.",
    )

    parser.add_argument(
        "--onnx-int8",
        required=True,
        help="Path to the dynamically quantized INT8 ONNX model.",
    )

    parser.add_argument(
        "--bench",
        default=DEFAULT_BENCH,
        help="Production benchmark mix .npy file.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help="Pinned CPU thread count.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Number of warm-up requests.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of benchmark samples; 0 means all.",
    )

    parser.add_argument(
        "--benchmarks-md",
        default="BENCHMARKS.md",
        help="Markdown file to append measured results to.",
    )

    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Print results without appending BENCHMARKS.md.",
    )

    return parser.parse_args()


def pin_threads(threads):
    value = str(threads)

    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def load_texts(path, samples=0):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark mix not found: {path}"
        )

    texts = np.load(
        path,
        allow_pickle=False,
    ).tolist()

    if samples > 0:
        texts = texts[:samples]

    if not texts:
        raise ValueError(
            "Benchmark mix contains no samples."
        )

    return texts


def summarise(name, latencies):
    values = np.asarray(
        latencies,
        dtype=np.float64,
    )

    return {
        "configuration": name,
        "samples": int(len(values)),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def benchmark(
    infer_one,
    texts,
    *,
    name,
    warmup=20,
):
    warmup_count = min(
        warmup,
        len(texts),
    )

    print(
        f"\nWarm-up: {name} "
        f"({warmup_count} requests)"
    )

    for text in texts[:warmup_count]:
        infer_one(text)

    print(f"Benchmarking: {name}")

    latencies = []

    for index, text in enumerate(
        texts,
        start=1,
    ):
        elapsed_ms = infer_one(text)
        latencies.append(elapsed_ms)

        if (
            index % 250 == 0
            or index == len(texts)
        ):
            print(
                f"Completed "
                f"{index}/{len(texts)}"
            )

    return summarise(
        name,
        latencies,
    )


def run_pytorch(
    model_path,
    tokenizer,
    texts,
    *,
    threads,
    max_length,
    padded,
    name,
    warmup,
):
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.set_num_threads(threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(f"\nLoading PyTorch model for: {name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model.to("cpu")
    model.eval()

    def infer_one(text):
        if padded:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        else:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

        start = time.perf_counter()

        with torch.inference_mode():
            model(**encoded)

        return (
            time.perf_counter() - start
        ) * 1000.0

    result = benchmark(
        infer_one,
        texts,
        name=name,
        warmup=warmup,
    )

    del model
    gc.collect()

    return result


def make_onnx_session(
    model_path,
    threads,
):
    import onnxruntime as ort

    options = ort.SessionOptions()

    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_onnx(
    model_path,
    tokenizer,
    texts,
    *,
    threads,
    name,
    warmup,
):
    print(f"\nLoading ONNX model for: {name}")

    session = make_onnx_session(
        model_path,
        threads,
    )

    input_names = {
        item.name
        for item in session.get_inputs()
    }

    def infer_one(text):
        encoded = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        inputs = {
            key: value
            for key, value in encoded.items()
            if key in input_names
        }

        start = time.perf_counter()

        session.run(
            None,
            inputs,
        )

        return (
            time.perf_counter() - start
        ) * 1000.0

    result = benchmark(
        infer_one,
        texts,
        name=name,
        warmup=warmup,
    )

    del session
    gc.collect()

    return result


def print_results(results, threads):
    print()
    print("=" * 86)
    print("BAYAN LAB 7 — CPU OPTIMISATION LADDER")
    print("=" * 86)

    print(
        f"{'Configuration':<38}"
        f"{'Mean (ms)':>12}"
        f"{'p50 (ms)':>12}"
        f"{'p99 (ms)':>12}"
    )

    print("-" * 86)

    for row in results:
        print(
            f"{row['configuration']:<38}"
            f"{row['mean_ms']:>12.2f}"
            f"{row['p50_ms']:>12.2f}"
            f"{row['p99_ms']:>12.2f}"
        )

    print("-" * 86)
    print(f"CPU threads: {threads}")


def append_markdown(
    path,
    results,
    *,
    threads,
    bench_path,
):
    path = Path(path)

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    lines = [
        "",
        "## Lab 7 — CPU inference optimisation ladder",
        "",
        f"- Run timestamp: `{timestamp}`",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- CPU threads: **{threads}**",
        f"- Benchmark mix: `{bench_path}`",
        (
            "- Timing scope: **model inference only** "
            "(model loading, tokenization and warm-up excluded)."
        ),
        "",
        "| Configuration | Samples | Mean (ms) | p50 (ms) | p99 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in results:
        lines.append(
            "| "
            f"{row['configuration']} | "
            f"{row['samples']} | "
            f"{row['mean_ms']:.2f} | "
            f"{row['p50_ms']:.2f} | "
            f"{row['p99_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            (
                "Lower latency is better. "
                "Production artefact selection must also "
                "pass correctness/parity checks."
            ),
            "",
        ]
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))

    print(f"\nAppended measured results to: {path}")


def main():
    args = parse_args()

    if args.threads < 1:
        raise ValueError(
            "--threads must be at least 1."
        )

    pin_threads(args.threads)

    from transformers import AutoTokenizer

    tokenizer_dir = (
        args.tokenizer_dir
        or args.pytorch_model
    )

    print("=" * 70)
    print("BAYAN LAB 7 — CPU INFERENCE BENCHMARK")
    print("=" * 70)
    print("CUDA is not used by this benchmark.")
    print("Threads:", args.threads)

    texts = load_texts(
        args.bench,
        args.samples,
    )

    print("Samples:", len(texts))
    print("Loading tokenizer:", tokenizer_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )

    results = []

    results.append(
        run_pytorch(
            args.pytorch_model,
            tokenizer,
            texts,
            threads=args.threads,
            max_length=512,
            padded=True,
            name="PyTorch FP32 padded 512",
            warmup=args.warmup,
        )
    )

    results.append(
        run_pytorch(
            args.pytorch_model,
            tokenizer,
            texts,
            threads=args.threads,
            max_length=128,
            padded=False,
            name="PyTorch FP32 dynamic <=128",
            warmup=args.warmup,
        )
    )

    results.append(
        run_onnx(
            args.onnx_fp32,
            tokenizer,
            texts,
            threads=args.threads,
            name="ONNX FP32 dynamic <=128",
            warmup=args.warmup,
        )
    )

    results.append(
        run_onnx(
            args.onnx_int8,
            tokenizer,
            texts,
            threads=args.threads,
            name="ONNX INT8 dynamic <=128",
            warmup=args.warmup,
        )
    )

    print_results(
        results,
        args.threads,
    )

    if not args.no_append:
        append_markdown(
            args.benchmarks_md,
            results,
            threads=args.threads,
            bench_path=args.bench,
        )


if __name__ == "__main__":
    main()
