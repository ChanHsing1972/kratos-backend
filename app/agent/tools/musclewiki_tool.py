import json
from typing import Any, Literal
from urllib import error, parse, request

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.agent.tools.http_utils import redact_url


ALLOWED_RESOURCES = {
    "root",
    "health",
    "statistics",
    "categories",
    "muscles",
    "filters",
    "exercises",
    "search",
    "random",
    "routines",
    "workouts",
}

ALLOWED_SUBRESOURCES = {
    "videos",
    "full",
}


class MuscleWikiInput(BaseModel):
    resource: Literal[
        "root",
        "health",
        "statistics",
        "categories",
        "muscles",
        "filters",
        "exercises",
        "search",
        "random",
        "routines",
        "workouts",
    ] = Field(
        default="exercises",
        description=(
            "MuscleWiki API resource to query. Use 'exercises' for exercise list or exercise details, "
            "'search' for text search, 'random' for random exercise, 'routines' and 'workouts' for training data, "
            "or metadata endpoints like 'health', 'statistics', 'categories', 'muscles', 'filters', 'root'."
        ),
    )
    item_id: int | None = Field(
        default=None,
        description=(
            "Optional resource ID. For example, use item_id=1 with resource='exercises' to request /exercises/1, "
            "or with resource='routines' to request /routines/1."
        ),
    )
    subresource: Literal["videos", "full"] | None = Field(
        default=None,
        description=(
            "Optional subresource. Supported combinations: exercises/{id}/videos, routines/{id}/full, workouts/{id}/full."
        ),
    )
    query_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional query string parameters. Examples: {'limit': 10, 'category': 'barbell'}, "
            "{'q': 'curl', 'difficulty': 'intermediate'}, {'detail': True, 'gender': 'male'}."
        ),
    )

    @model_validator(mode="after")
    def validate_resource_shape(self):
        if self.resource not in ALLOWED_RESOURCES:
            raise ValueError(f"Unsupported resource: {self.resource}")

        if self.subresource and self.subresource not in ALLOWED_SUBRESOURCES:
            raise ValueError(f"Unsupported subresource: {self.subresource}")

        if self.item_id is None and self.subresource is not None:
            raise ValueError("subresource requires item_id")

        if self.resource in {"root", "health", "statistics", "categories", "muscles", "filters", "search", "random"}:
            if self.item_id is not None:
                raise ValueError(f"resource '{self.resource}' does not support item_id")
            if self.subresource is not None:
                raise ValueError(f"resource '{self.resource}' does not support subresource")

        if self.resource == "search" and "q" not in self.query_params:
            raise ValueError("resource 'search' requires query_params['q']")

        if self.resource in {"routines", "workouts"} and self.subresource == "videos":
            raise ValueError(f"resource '{self.resource}' does not support subresource 'videos'")

        if self.resource == "exercises" and self.subresource == "full":
            raise ValueError("resource 'exercises' does not support subresource 'full'")

        if self.resource not in {"exercises", "routines", "workouts"} and self.item_id is not None:
            raise ValueError(f"resource '{self.resource}' does not support item_id")

        if self.resource in {"routines", "workouts"} and self.subresource and self.subresource != "full":
            raise ValueError(f"resource '{self.resource}' only supports subresource 'full'")

        return self


class MuscleWikiTool:
    name = "musclewiki_api"
    description = (
        "Read data from the MuscleWiki API using official documented endpoints. "
        "Supports root metadata, health, statistics, categories, muscles, filters, exercise list/detail/videos, "
        "text search, random exercise, routines, and workouts. Use query_params for filtering, pagination, and search."
    )
    args_schema = MuscleWikiInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = MuscleWikiInput(**args)

        if not settings.MUSCLEWIKI_API_KEY:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "MUSCLEWIKI_API_KEY is not configured.",
                "url": None,
                "data": None,
            }

        path = self._build_path(payload)
        query_string = self._build_query_string(payload.query_params)
        url = f"{settings.MUSCLEWIKI_API_BASE_URL.rstrip('/')}{path}{query_string}"

        req = request.Request(
            url=url,
            headers={
                "X-API-Key": settings.MUSCLEWIKI_API_KEY,
                "Accept": "application/json",
                "User-Agent": "Kratos-Agent/1.0",
            },
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=settings.MUSCLEWIKI_TIMEOUT_SECONDS) as response:
                raw_body = response.read().decode("utf-8")
                data = self._load_json_or_text(raw_body)
                return {
                    "ok": True,
                    "url": redact_url(url),
                    "resource": payload.resource,
                    "item_id": payload.item_id,
                    "subresource": payload.subresource,
                    "status_code": response.status,
                    "headers": dict(response.headers.items()),
                    "data": data,
                }
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            parsed_error = self._load_json_or_text(error_body)
            return {
                "ok": False,
                "url": redact_url(url),
                "resource": payload.resource,
                "item_id": payload.item_id,
                "subresource": payload.subresource,
                "status_code": exc.code,
                "error_type": self._classify_error(exc.code, parsed_error),
                "message": self._extract_error_message(parsed_error, exc.reason),
                "retryable": self._extract_retryable(parsed_error),
                "owner_action_required": self._extract_owner_action_required(parsed_error),
                "data": parsed_error,
            }
        except error.URLError as exc:
            return {
                "ok": False,
                "url": redact_url(url),
                "resource": payload.resource,
                "item_id": payload.item_id,
                "subresource": payload.subresource,
                "status_code": None,
                "error_type": "network_error",
                "message": f"MuscleWiki API request failed: {exc.reason}",
                "retryable": True,
                "owner_action_required": False,
                "data": None,
            }

    @staticmethod
    def _build_path(payload: MuscleWikiInput) -> str:
        if payload.resource == "root":
            return "/"

        path = f"/{payload.resource}"
        if payload.item_id is not None:
            path += f"/{payload.item_id}"
        if payload.subresource is not None:
            path += f"/{payload.subresource}"
        return path

    @staticmethod
    def _build_query_string(query_params: dict[str, Any]) -> str:
        if not query_params:
            return ""
        normalized: dict[str, Any] = {}
        for key, value in query_params.items():
            if value is None:
                continue
            normalized[key] = str(value).lower() if isinstance(value, bool) else value
        if not normalized:
            return ""
        return f"?{parse.urlencode(normalized, doseq=True)}"

    @staticmethod
    def _load_json_or_text(raw_body: str) -> Any:
        if not raw_body:
            return None
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body

    @staticmethod
    def _classify_error(status_code: int, parsed_error: Any) -> str:
        if isinstance(parsed_error, dict):
            detail = str(parsed_error.get("detail") or "").lower()
            error_name = str(parsed_error.get("error_name") or "").lower()
            message = str(parsed_error.get("message") or "").lower()
            if status_code == 403 and "playground" in detail + message:
                return "subscription_restricted"
            if status_code == 403 and error_name:
                return error_name
            if status_code == 401:
                return "authentication_error"
            if status_code == 404:
                return "not_found"
            if status_code == 422:
                return "validation_error"
            if status_code == 429:
                return "rate_limit_exceeded"
        return f"http_{status_code}"

    @staticmethod
    def _extract_error_message(parsed_error: Any, fallback_reason: str) -> str:
        if isinstance(parsed_error, dict):
            for key in ("message", "detail", "title", "what_you_should_do"):
                value = parsed_error.get(key)
                if value:
                    return str(value)
        if isinstance(parsed_error, str) and parsed_error.strip():
            return parsed_error
        return str(fallback_reason)

    @staticmethod
    def _extract_retryable(parsed_error: Any) -> bool | None:
        if isinstance(parsed_error, dict) and "retryable" in parsed_error:
            value = parsed_error.get("retryable")
            if isinstance(value, bool):
                return value
        return None

    @staticmethod
    def _extract_owner_action_required(parsed_error: Any) -> bool | None:
        if isinstance(parsed_error, dict) and "owner_action_required" in parsed_error:
            value = parsed_error.get("owner_action_required")
            if isinstance(value, bool):
                return value
        return None



def get_musclewiki_tool():
    return MuscleWikiTool()
