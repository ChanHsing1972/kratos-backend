"""
FastAPI 应用入口文件
"""

import uvicorn
from fastapi import FastAPI
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Kratos Fitness Agent Backend",
)

@app.get("/")
def root():
    return {"message": "Welcome to Kratos Fitness Agent API."}


@app.get("/health")
def health():
    """Health check: returns service status and database connectivity."""
    db_ok = False
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    finally:
        try:
            db.close()
        except Exception:
            pass

    return {"status": "ok", "db": db_ok}


# Include routers here
# app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
