from fastapi import APIRouter

from app.schemas.prediction import PredictionResponse
from app.services.gesture_service import GestureService

router = APIRouter()

# One shared instance for now - in Week 3, this is where a trained
# model would be loaded once and reused across requests.
gesture_service = GestureService()


@router.post("/predict", response_model=PredictionResponse)
def predict():
    """
    Returns a gesture prediction. Currently returns dummy data from
    GestureService - the router itself contains no prediction logic,
    it only delegates.
    """
    return gesture_service.predict()