
"""Bayan FastAPI service for Lab 7 and the capstone."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bayan.serving.classifier import TopicClassifier


class ClassifyRequest(BaseModel):
    """Input contract for topic classification."""

    text: str = Field(
        ...,
        min_length=1,
        description="Arabic or English citizen-feedback text.",
    )


class ClassifyResponse(BaseModel):
    """Output contract for topic classification."""

    topic: str
    class_id: int
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Lab 7 classifier once and run a startup canary."""

    classifier = TopicClassifier(
        threads=4,
    )

    canary = classifier.classify(
        "إنارة الشارع لا تعمل"
    )

    if not canary.get("topic"):
        raise RuntimeError(
            "Bayan classifier startup canary failed."
        )

    app.state.classifier = classifier
    app.state.canary = canary

    yield


app = FastAPI(
    title="Bayan — Bilingual Citizen-Feedback Intelligence Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Service readiness endpoint."""

    return {
        "status": "ready",
        "model": "ONNX INT8 topic classifier",
        "threads": 4,
    }


@app.post(
    "/v1/classify",
    response_model=ClassifyResponse,
)
def classify(payload: ClassifyRequest):
    """Classify one Arabic or English feedback message."""

    try:
        return app.state.classifier.classify(
            payload.text
        )

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/v1/entities")
def entities(payload: dict):
    """Capstone placeholder for NER."""

    raise HTTPException(
        status_code=501,
        detail="Capstone NER endpoint not implemented yet.",
    )


@app.post("/v1/search")
def search(payload: dict):
    """Capstone placeholder for semantic search."""

    raise HTTPException(
        status_code=501,
        detail="Capstone search endpoint not implemented yet.",
    )


@app.post("/v1/analyse")
def analyse(payload: dict):
    """Capstone placeholder for the integrated pipeline."""

    raise HTTPException(
        status_code=501,
        detail="Capstone analyse endpoint not implemented yet.",
    )
