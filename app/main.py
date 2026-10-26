from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.database import SessionLocal
from app.routers.dashboard import router as dashboard_router
from app.routers.learning_events import router as learning_events_router
from app.routers.materials import router as materials_router
from app.routers.rag_chatbot import router as rag_chatbot_router
from app.routers.recommendations import router as recommendations_router
from app.routers.revision_schedules import router as revision_schedules_router
from app.routers.students import router as students_router
from app.routers.topic_recall import router as topic_recall_router
from app.routers.topics import router as topics_router

# The old predictions router is intentionally NOT mounted: its student-level
# GET endpoints do not check who is logged in, so any caller could read any
# student's recall history. Mount it again only after auth checks are added.

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for the Smart Study Recall learning platform.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

app.include_router(students_router)
app.include_router(topics_router)
app.include_router(learning_events_router)
app.include_router(revision_schedules_router)
app.include_router(recommendations_router)
app.include_router(dashboard_router)
app.include_router(topic_recall_router)
app.include_router(materials_router)
app.include_router(rag_chatbot_router)


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
