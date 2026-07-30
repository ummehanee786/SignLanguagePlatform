"""
gesture_service.py

This service sits between the API router and the actual AI logic.
The router should never contain prediction logic itself - it just
calls this service and returns whatever it gets back.

predict() now calls the real AI module (app.ai.ml.inference.engine) -
this is the one file that had to change to go from dummy data to a
real trained model. The router (app/api/predict.py) and the response
schema's shape didn't need to change for this to happen; only new,
optional fields were added to represent everyday failure cases (e.g.
no hand detected) that dummy data never had to account for.
"""

import time

from app.ai.ml.inference.engine import predict_from_bytes
from app.ai.ml.inference.result import PredictionResult
from app.schemas.prediction import PredictionResponse


class GestureService:
    def __init__(self):
        # The AI engine (model + detector) is loaded lazily on first
        # use via engine.get_engine() - not here - so importing this
        # service (e.g. at app startup) never requires a trained model
        # to already exist on disk. The first prediction request pays
        # the one-time load cost; every request after that reuses it.
        pass

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """
        Runs a real prediction on raw image bytes (e.g. the contents of
        an uploaded file) and maps the AI module's PredictionResult
        into this service's API-facing PredictionResponse schema.

        This method deliberately never raises for expected failure
        cases (no hand detected, bad image data) - engine.predict_from_bytes
        already guarantees that; it always returns a PredictionResult,
        which we convert into a PredictionResponse with success=False
        instead of an HTTP error, so the client gets a normal response
        it can render (e.g. "no hand detected, try again") rather than
        having to special-case error responses.
        """
        result: PredictionResult = predict_from_bytes(image_bytes)
        return self._to_response(result)

    @staticmethod
    def _to_response(result: PredictionResult) -> PredictionResponse:
        return PredictionResponse(
            success=result.success,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            above_confidence_threshold=result.above_confidence_threshold,
            probabilities=result.probabilities,
            error=result.error,
            model_version=result.model_version,
            processing_time=(result.total_time_ms or 0.0) / 1000.0,  # ms -> seconds
            landmarks=result.landmarks,
            has_person=result.has_person,
            hand_count=result.hand_count,
            upper_body_visible=result.upper_body_visible,
            partial_hand_visible=result.partial_hand_visible,
            hand_centered=result.hand_centered,
        )