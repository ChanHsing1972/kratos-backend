import logging

from app.db.session import Base, SessionLocal, engine
from app.models import ExerciseLibraryItem  # noqa: F401 - imported so create_all sees the model.
from app.services.exercise_library import sync_rapidapi_exercise_library


logger = logging.getLogger(__name__)


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        result = sync_rapidapi_exercise_library(db)
        logger.info("Exercise library sync result: %s", result)
    finally:
        db.close()


if __name__ == "__main__":
    main()
