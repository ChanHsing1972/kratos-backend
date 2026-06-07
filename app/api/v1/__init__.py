from fastapi import APIRouter

from app.api.v1.endpoints.agent_tool import router as agent_tool_router
from app.api.v1.endpoints.agent_checkin import router as agent_checkin_router
from app.api.v1.endpoints.agent_chat import router as agent_chat_router
from app.api.v1.endpoints.agent_run import router as agent_run_router
from app.api.v1.endpoints.apple_health import router as apple_health_router
from app.api.v1.endpoints.conversation_session import router as conversation_session_router
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.body_metric import router as body_metric_router
from app.api.v1.endpoints.diet import router as diet_router
from app.api.v1.endpoints.health_metric import router as health_metric_router
from app.api.v1.endpoints.integrations import router as integrations_router
from app.api.v1.endpoints.profile import router as profile_router
from app.api.v1.endpoints.skill import router as skill_router
from app.api.v1.endpoints.training_plan import router as training_plan_router
from app.api.v1.endpoints.upload import router as upload_router
from app.api.v1.endpoints.workout_session import router as workout_session_router
from app.api.v1.endpoints.workout_log import router as workout_log_router

api_router = APIRouter()
api_router.include_router(agent_chat_router, prefix="/agent", tags=["agent"])
api_router.include_router(agent_tool_router, prefix="/tools", tags=["tools"])
api_router.include_router(agent_run_router, prefix="/agent", tags=["agent-runs"])
api_router.include_router(apple_health_router, prefix="/apple-health", tags=["apple-health"])
api_router.include_router(conversation_session_router, prefix="/agent", tags=["agent-sessions"])
api_router.include_router(agent_checkin_router, prefix="/agent-checkins", tags=["agent-checkins"])
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(body_metric_router, prefix="/body-metrics", tags=["body-metrics"])
api_router.include_router(diet_router, prefix="/diet", tags=["diet"])
api_router.include_router(health_metric_router, prefix="/health-metrics", tags=["health-metrics"])
api_router.include_router(integrations_router, prefix="/integrations", tags=["integrations"])
api_router.include_router(profile_router, prefix="/profile", tags=["profile"])
api_router.include_router(skill_router, prefix="/skills", tags=["skills"])
api_router.include_router(training_plan_router, prefix="/plans", tags=["plans"])
api_router.include_router(upload_router, prefix="/uploads", tags=["uploads"])
api_router.include_router(workout_session_router, prefix="/workout-sessions", tags=["workout-sessions"])
api_router.include_router(workout_log_router, prefix="/workout-logs", tags=["workout-logs"])
