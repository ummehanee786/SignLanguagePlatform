from typing import Dict, List, Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """
    Response returned by the /predict endpoint.

    Note: predicted_class/confidence are now Optional. A real system has
    to represent "no hand detected" and similar everyday failures, not
    just successful predictions - success=False + error carries that
    case instead of the endpoint raising an error for something that
    isn't really exceptional (a user just hasn't shown a hand yet).

    success                    : whether a prediction was made at all
    predicted_class            : the gesture/letter predicted (e.g. "A") - None on failure
    confidence                 : model's confidence in that prediction (0.0-1.0) - None on failure
    above_confidence_threshold : whether confidence cleared the engine's configured
                                  threshold - lets the caller distinguish "predicted,
                                  but not confidently enough to act on" from a solid hit
    probabilities               : full class -> probability distribution, useful for
                                  showing top-N alternatives or debugging
    error                       : human-readable reason when success=False
    model_version               : which model version served this prediction, for
                                  monitoring/debugging (see app/ai/ml/inference/model_loader.py)
    processing_time             : total pipeline time, in seconds
    """
    success: bool = True
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    above_confidence_threshold: Optional[bool] = None
    probabilities: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    model_version: Optional[str] = None
    processing_time: float
    # Raw 21-landmark flat array (63 floats) from MediaPipe.
    # Populated on successful predictions; None when no hand was detected.
    landmarks: Optional[List[float]] = None