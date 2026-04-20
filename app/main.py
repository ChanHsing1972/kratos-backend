"""
FastAPI 应用入口文件
"""

import uvicorn
from fastapi import FastAPI
from sqlalchemy import text

from app.api.v1 import api_router
from app.core.config import settings
from app.db.session import Base, SessionLocal, engine
from app.models import User


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Kratos Fitness Agent Backend",
)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {"message": "Welcome to Kratos Fitness Agent API."}


@app.get("/health")
def health():
    """Health check: returns service status and database connectivity."""
    db_ok = False
    database_uri = (
        f"mysql+pymysql://{settings.MYSQL_USER}:***@"
        f"{settings.MYSQL_SERVER}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
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
                "host": settings.MYSQL_SERVER,
                "port": settings.MYSQL_PORT,
                "name": settings.MYSQL_DB,
                "user": settings.MYSQL_USER,
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
            "host": settings.MYSQL_SERVER,
            "port": settings.MYSQL_PORT,
            "name": settings.MYSQL_DB,
            "user": settings.MYSQL_USER,
            "uri": database_uri,
        },
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
