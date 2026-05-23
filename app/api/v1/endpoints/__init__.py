from app.api.v1.endpoints.agent_tool import router as agent_tool_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.profile import router as profile_router
from app.api.v1.endpoints.training_plan import router as training_plan_router
from app.api.v1.endpoints.workout_log import router as workout_log_router

__all__ = [
    "agent_tool_router",
    "auth_router",
    "profile_router",
    "training_plan_router",
    "workout_log_router",
]
