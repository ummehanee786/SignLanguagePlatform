from typing import Optional

from fastapi import APIRouter

from app.schemas.preprocess import PreprocessResponse
from app.services.preprocessing_service import PreprocessingService

router = APIRouter()

preprocessing_service = PreprocessingService()


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess(limit: Optional[int] = None, normalize: bool = False):
    """
    Triggers dataset preprocessing (landmark extraction -> CSV ->
    validation report) and returns a summary.

    IMPORTANT: without `limit`, this processes the ENTIRE dataset
    (87,000+ images), which takes well over an hour. Use the `limit`
    query parameter (e.g. POST /preprocess?limit=5) to test quickly
    on a small subset first.
    """
    result = preprocessing_service.run(limit_per_class=limit, normalize=normalize)

    return PreprocessResponse(
        success=True,
        message="Dataset preprocessing completed.",
        data=result,
    )