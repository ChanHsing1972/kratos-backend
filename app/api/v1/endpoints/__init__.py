from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.profile import router as profile_router
from app.api.v1.endpoints.training_plan import router as training_plan_router
from app.api.v1.endpoints.workout_log import router as workout_log_router

__all__ = [
    "auth_router",
    "profile_router",
    "training_plan_router",
    "workout_log_router",
]
