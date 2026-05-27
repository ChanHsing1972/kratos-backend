from app.db.session import Base, SessionLocal, engine
from app.models import ExerciseLibraryItem  # noqa: F401 - imported so create_all sees the model.
from app.services.exercise_library import sync_rapidapi_exercise_library


def main() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        result = sync_rapidapi_exercise_library(db)
        print(result)
    finally:
        db.close()


if __name__ == "__main__":
    main()
