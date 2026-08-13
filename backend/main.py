from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.lessons import router as lessons_router
from app.api.practice import router as practice_router
from app.api.reports import router as reports_router
from app.api.admin import router as admin_router
from app.api.certifications import router as certifications_router
from app.api.notifications import router as notifications_router

logger = setup_logging()

app = FastAPI(title=settings.app_name, debug=settings.debug)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": True, "message": "Validation Error", "details": exc.errors()}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": "Internal Server Error", "details": str(exc) if settings.debug else "An unexpected error occurred."}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(lessons_router)
app.include_router(practice_router)
app.include_router(reports_router)
app.include_router(admin_router)
app.include_router(certifications_router)
app.include_router(notifications_router)

logger.info("FastAPI app initialized with all routers.")
