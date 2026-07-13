from fastapi import FastAPI

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.lessons import router as lessons_router
from app.api.practice import router as practice_router
from app.api.preprocess import router as preprocess_router

logger = setup_logging()

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(lessons_router)
app.include_router(practice_router)
app.include_router(preprocess_router)

logger.info("FastAPI app initialized with all routers.")