from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """
    Response returned by the /predict endpoint.

    predicted_class : the gesture/letter the model predicts (e.g. "A")
    confidence       : how confident the model is in that prediction (0.0-1.0)
    processing_time  : how long inference took, in seconds - will be used
                        later to monitor model performance / latency
    """
    predicted_class: str
    confidence: float
    processing_time: float