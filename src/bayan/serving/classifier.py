
"""Reusable ONNX INT8 topic-classification service for Bayan."""

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from bayan.preprocessing.core import preprocess


DEFAULT_ARTIFACT_DIR = Path(
    "artifacts/serving/topic_classifier_int8"
)


class TopicClassifier:
    """Load and serve the Lab 7 INT8 topic classifier."""

    def __init__(
        self,
        artifact_dir=None,
        threads=4,
    ):
        self.artifact_dir = Path(
            artifact_dir
            or os.getenv(
                "BAYAN_CLASSIFIER_DIR",
                str(DEFAULT_ARTIFACT_DIR),
            )
        )

        self.model_path = (
            self.artifact_dir / "model.onnx"
        )

        self.config_path = (
            self.artifact_dir / "config.json"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {self.config_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.artifact_dir),
            local_files_only=True,
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        raw_id2label = config.get(
            "id2label",
            {}
        )

        self.id2label = {
            int(key): value
            for key, value in raw_id2label.items()
        }

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_names = {
            item.name
            for item in self.session.get_inputs()
        }

    def classify(self, text):
        """Preprocess and classify one feedback text."""

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text must not be empty"
            )

        cleaned_text = preprocess(text)

        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        inputs = {
            key: value
            for key, value in encoded.items()
            if key in self.input_names
        }

        logits = self.session.run(
            None,
            inputs,
        )[0][0]

        predicted_id = int(
            np.argmax(logits)
        )

        shifted = logits - np.max(logits)

        probabilities = (
            np.exp(shifted)
            / np.exp(shifted).sum()
        )

        confidence = float(
            probabilities[predicted_id]
        )

        label = self.id2label.get(
            predicted_id,
            str(predicted_id),
        )

        return {
            "topic": label,
            "class_id": predicted_id,
            "confidence": round(
                confidence,
                6,
            ),
        }
