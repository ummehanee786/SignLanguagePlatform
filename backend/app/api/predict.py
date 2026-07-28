from fastapi import APIRouter, File, UploadFile

from app.schemas.prediction import PredictionResponse
from app.services.gesture_service import GestureService

router = APIRouter()

# One shared instance - the AI model itself is loaded lazily, on the
# first prediction request, and reused for every request after that
# (see GestureService / app.ai.ml.inference.engine.get_engine()).
gesture_service = GestureService()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Accepts an uploaded image (e.g. a webcam frame) and returns a
    gesture prediction. The router contains no prediction logic itself -
    it just reads the raw bytes and delegates to GestureService, which
    delegates to the AI module. Note there's no cv2/numpy/mediapipe
    import anywhere in this file: decoding image bytes into a usable
    frame happens inside app/ai/handtracking/detector.py, the one file
    in the backend allowed to touch those libraries.
    """
    image_bytes = await file.read()
    return gesture_service.predict(image_bytes)