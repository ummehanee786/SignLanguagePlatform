"""
result.py

The "Prediction Object" the whole pipeline builds towards:

    ... -> Confidence Threshold -> Prediction Object

PredictionResult is a plain dataclass, not a Pydantic model. That's
deliberate: this is the AI module's own internal contract, independent of
FastAPI/Pydantic. The rest of the app (gesture_service.py, API schemas)
can convert this into whatever response shape it needs - but this module
itself doesn't depend on the web framework to describe its own output.

Two ways to end up with a PredictionResult:
  - success (a hand was found and the model produced a class + confidence)
  - failure (no hand detected, or the landmarks failed validation) -
    engine.py never raises for these expected, everyday cases; it always
    returns a PredictionResult so callers have one consistent thing to
    check (`result.success`) instead of needing try/except around every
    call.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class PredictionResult:
    success: bool

    # Populated when success=True
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    above_confidence_threshold: Optional[bool] = None
    probabilities: Optional[Dict[str, float]] = None

    # Populated when success=False
    error: Optional[str] = None

    # Raw 21-landmark flat array (63 floats: x0,y0,z0 … x20,y20,z20) from
    # MediaPipe, passed through so downstream consumers (e.g. the Feedback
    # Engine) can inspect hand shape without re-running detection.
    landmarks: Optional[List[float]] = None

    # Always populated - inference metadata, useful for logging/monitoring
    # regardless of whether the prediction itself succeeded.
    model_version: Optional[str] = None
    model_inference_time_ms: Optional[float] = None  # model.predict_proba() call only
    total_time_ms: Optional[float] = None  # full pipeline: landmarks -> ... -> result
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """JSON-serializable representation, e.g. for logging or an API response."""
        return asdict(self)

    @classmethod
    def failure(
        cls,
        error: str,
        model_version: Optional[str] = None,
        total_time_ms: Optional[float] = None,
    ) -> "PredictionResult":
        return cls(
            success=False,
            error=error,
            model_version=model_version,
            total_time_ms=total_time_ms,
        )

    @classmethod
    def ok(
        cls,
        predicted_class: str,
        confidence: float,
        above_confidence_threshold: bool,
        probabilities: Dict[str, float],
        model_version: str,
        model_inference_time_ms: float,
        total_time_ms: float,
        landmarks: Optional[List[float]] = None,
    ) -> "PredictionResult":
        return cls(
            success=True,
            predicted_class=predicted_class,
            confidence=confidence,
            above_confidence_threshold=above_confidence_threshold,
            probabilities=probabilities,
            model_version=model_version,
            model_inference_time_ms=model_inference_time_ms,
            total_time_ms=total_time_ms,
            landmarks=landmarks,
        )