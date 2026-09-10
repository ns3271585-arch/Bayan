"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.preprocessing.core import PREPROC_VERSION, preprocess


DEFAULT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CaseSearch:
    def __init__(self, prefix: str):
        """Load and validate the persisted search artifacts."""

        self.prefix = Path(prefix)

        self.manifest_path = Path(f"{prefix}_manifest.json")
        self.index_path = Path(f"{prefix}_index.faiss")
        self.metadata_path = Path(f"{prefix}_metadata.parquet")

        for path in [
            self.manifest_path,
            self.index_path,
            self.metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing search artifact: {path}")

        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )

        required_keys = {
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        }
        missing = required_keys - self.manifest.keys()
        if missing:
            raise ValueError(
                f"Manifest missing required fields: {sorted(missing)}"
            )

        # Prevent serving an index built with a different preprocessing contract.
        if self.manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError(
                "Preprocessing version mismatch: "
                f"index={self.manifest['preproc_version']} "
                f"runtime={PREPROC_VERSION}"
            )

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.metadata_path)

        # Manifest/index/metadata integrity checks.
        if self.index.ntotal != int(self.manifest["n_vectors"]):
            raise ValueError("FAISS vector count does not match manifest")

        if self.index.d != int(self.manifest["dim"]):
            raise ValueError("FAISS dimension does not match manifest")

        if len(self.metadata) != int(self.manifest["n_vectors"]):
            raise ValueError("Metadata row count does not match manifest")

        if not bool(self.manifest.get("l2_normalized", False)):
            raise ValueError("Index manifest does not guarantee L2 normalisation")

        # Load the exact encoder pinned when the index was built.
        self.encoder = SentenceTransformer(self.manifest["model"])

        # Multilingual second-stage reranker.
        self.reranker_name = DEFAULT_RERANKER
        self.reranker = CrossEncoder(self.reranker_name)

    def _retrieve(self, query: str, candidates: int = 50):
        """Return first-stage bi-encoder candidates."""

        if candidates <= 0:
            raise ValueError("candidates must be positive")

        normalized_query = preprocess(query)

        query_vector = self.encoder.encode(
            [normalized_query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype="float32")

        # Must match the L2-normalised corpus index.
        faiss.normalize_L2(query_vector)

        n_candidates = min(candidates, self.index.ntotal)

        scores, indices = self.index.search(
            query_vector,
            n_candidates,
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            row = self.metadata.iloc[int(idx)]

            results.append(
                {
                    "case_id": str(row["case_id"]),
                    "lang": str(row["lang"]),
                    "topic": str(row["topic"]),
                    "case_text": str(row["case_text"]),
                    "indexed_text": str(row["indexed_text"]),
                    "resolution": str(row["resolution"]),
                    "bi_score": float(score),
                }
            )

        return normalized_query, results

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = -3.0,
        rerank: bool = True,
    ):
        """Search cases using bi-encoder retrieval and optional CE reranking."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            return []

        if k <= 0:
            raise ValueError("k must be positive")

        normalized_query, results = self._retrieve(
            query,
            candidates=candidates,
        )

        if not results:
            return []

        if not rerank:
            return results[:k]

        pairs = [
             (normalized_query, item["indexed_text"])

            for item in results
        ]

        ce_scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        for item, score in zip(results, ce_scores):
            item["score"] = float(score)

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # Honest no-result behaviour.
        filtered = [
            item
            for item in results
            if item["score"] >= min_score
        ]

        return filtered[:k]