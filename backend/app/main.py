
from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.routers.students import router as students_router
from app.config import settings
from app.database import SessionLocal
from app.routers.topics import router as topics_router
from app.routers.learning_events import (
    router as learning_events_router,
)
from app.routers.revision_schedules import (
    router as revision_schedules_router,
)
from app.routers.predictions import router as predictions_router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for the Smart Study Recall learning platform.",
)

app.include_router(students_router)
app.include_router(topics_router)
app.include_router(learning_events_router)
app.include_router(revision_schedules_router)
app.include_router(predictions_router)
@app.get("/")
def root():
    return {
        "message": "Welcome to Smart Study Recall",
        "api_docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "main-backend",
        "ml_service_url": settings.ml_service_url,
    }


@app.get("/health/database")
def database_health():
    if SessionLocal is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not configured.",
        )

    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "database": "connected",
        }

    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail="Database connection failed.",
        )