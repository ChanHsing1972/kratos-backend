"""
FastAPI 应用入口文件
"""

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1 import api_router
from app.core.config import settings
from app.services.upload import UPLOAD_DIR
from app.db.schema_sync import ensure_runtime_schema
from app.db.session import Base, SessionLocal, engine
from app.models import User
from app.services.conversation_session import backfill_conversation_sessions_from_agent_runs
from app.services.skill import seed_builtin_skills


from fastapi.middleware.cors import CORSMiddleware

try:
    Base.metadata.create_all(bind=engine)
    ensure_runtime_schema(engine)
except SQLAlchemyError as exc:
    print(f"[startup] Database initialization skipped: {exc}")
try:
    with SessionLocal() as seed_db:
        seed_builtin_skills(seed_db)
        backfill_conversation_sessions_from_agent_runs(seed_db)
except SQLAlchemyError as exc:
    print(f"[startup] Database seed/backfill skipped: {exc}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Kratos Fitness Agent Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGIN_LIST,
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def root():
    return {"message": "Welcome to Kratos Fitness Agent API."}


@app.get("/health")
def health():
    """Health check: returns service status and database connectivity."""
    db_ok = False
    database_uri = (
        f"postgresql+psycopg2://{settings.PG_USER}:***@"
        f"{settings.PG_SERVER}:{settings.PG_PORT}/{settings.PG_DB}"
    )
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        return {
            "status": "error",
            "db": False,
            "database": {
                "host": settings.PG_SERVER,
                "port": settings.PG_PORT,
                "name": settings.PG_DB,
                "user": settings.PG_USER,
                "uri": database_uri,
            },
            "detail": str(exc),
        }
    finally:
        try:
            db.close()
        except Exception:
            pass

    return {
        "status": "ok",
        "db": db_ok,
        "database": {
            "host": settings.PG_SERVER,
            "port": settings.PG_PORT,
            "name": settings.PG_DB,
            "user": settings.PG_USER,
            "uri": database_uri,
        },
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
