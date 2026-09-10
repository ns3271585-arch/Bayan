"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_index(
    prefix: str,
    limit: int | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Build and persist a versioned FAISS index over Bayan historical cases."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")

    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "data" / "search" / "bayan_cases.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Case corpus not found: {data_path}")

    # Load the historical case corpus.
    df = pd.read_csv(data_path)

    if limit is not None:
        df = df.head(limit).copy()
    else:
        df = df.copy()

    if df.empty:
        raise ValueError("No cases available to index")

    # Apply the same shared preprocessing contract used by training/evaluation.
    df["case_text"] = df["case_text"].fillna("").astype(str)
    df["indexed_text"] = df["case_text"].map(preprocess)

    # Encode the corpus with the bilingual/multilingual bi-encoder.
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        df["indexed_text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Encoder returned invalid embeddings")

    # Explicit L2 normalisation makes inner product equivalent to cosine similarity.
    faiss.normalize_L2(embeddings)

    n_vectors, dim = embeddings.shape

    # Build an exact cosine-similarity FAISS index.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = Path(f"{prefix}_index.faiss")
    metadata_path = Path(f"{prefix}_metadata.parquet")
    manifest_path = Path(f"{prefix}_manifest.json")

    # Persist FAISS index.
    faiss.write_index(index, str(index_path))

    # Persist metadata in exactly the same row order as the FAISS vectors.
    metadata_columns = [
        "case_id",
        "lang",
        "topic",
        "case_text",
        "indexed_text",
        "resolution",
        "status",
        "closed_at",
        "synthetic",
    ]
    df[metadata_columns].reset_index(drop=True).to_parquet(
        metadata_path,
        index=False,
    )

    # Persist the version/integrity manifest.
    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(n_vectors),
        "dim": int(dim),
        "index_type": "IndexFlatIP",
        "metric": "cosine",
        "l2_normalized": True,
        "text_field": "case_text",
        "index_file": index_path.name,
        "metadata_file": metadata_path.name,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest