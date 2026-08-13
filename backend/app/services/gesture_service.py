"""
gesture_service.py

This service sits between the API router and the actual AI logic.
The router should never contain prediction logic itself - it just
calls this service and returns whatever it gets back.

Right now predict() returns dummy values, since we don't have a
trained classifier yet (that's Week 3). Importantly, the ROUTER
doesn't know or care that this is dummy data - when the real model
is ready, only the inside of this class changes. The API contract
(the schema, the endpoint) stays exactly the same.
"""

import time

from app.schemas.prediction import PredictionResponse


class GestureService:
    def __init__(self):
        # In Week 3, this is where we'll load the trained classifier
        # (e.g. a saved model file) once, so it's ready to use for
        # every prediction instead of reloading it each request.
        pass

    def predict(self) -> PredictionResponse:
        """
        Returns a dummy prediction for now.

        Later, this will:
          1. Use HandLandmarkDetector (from app.ai.handtracking) to
             extract landmarks from an incoming image/frame.
          2. Feed those landmarks into the trained classifier.
          3. Return the real predicted class + confidence.

        The router calling this method won't need to change at all
        when that happens.
        """
        start_time = time.time()

        # --- Dummy stand-in values ---
        predicted_class = "A"
        confidence = 0.0

        processing_time = time.time() - start_time

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            processing_time=processing_time,
        )