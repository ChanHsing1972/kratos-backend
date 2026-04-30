import json
from typing import Any
from urllib import error, parse, request

from pydantic import BaseModel, Field

from app.core.config import settings
from app.agent.tools.http_utils import redact_url


class RapidApiBodypartsInput(BaseModel):
    query_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional query string parameters for the bodyparts endpoint.",
    )


class RapidApiBodypartsTool:
    name = "rapidapi_bodyparts"
    description = (
        "Fetch the available body parts list from the AscendAPI exercise database hosted on RapidAPI. "
        "Use this when you need canonical body part names for exercise filtering or recommendation."
    )
    args_schema = RapidApiBodypartsInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = RapidApiBodypartsInput(**args)

        if not settings.RAPIDAPI_KEY:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "RAPIDAPI_KEY is not configured.",
                "url": None,
                "data": None,
            }

        if not settings.RAPIDAPI_HOST:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "RAPIDAPI_HOST is not configured.",
                "url": None,
                "data": None,
            }

        query_string = self._build_query_string(payload.query_params)
        url = f"{settings.RAPIDAPI_BASE_URL.rstrip('/')}/api/v1/bodyparts{query_string}"

        req = request.Request(
            url=url,
            headers={
                "Content-Type": "application/json",
                "x-rapidapi-host": settings.RAPIDAPI_HOST,
                "x-rapidapi-key": settings.RAPIDAPI_KEY,
                "Accept": "application/json",
                "User-Agent": "Kratos-Agent/1.0",
            },
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=settings.RAPIDAPI_TIMEOUT_SECONDS) as response:
                raw_body = response.read().decode("utf-8", errors="ignore")
                data = self._load_json_or_text(raw_body)
                return {
                    "ok": True,
                    "url": redact_url(url),
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
                "status_code": exc.code,
                "error_type": self._classify_error(exc.code),
                "message": self._extract_error_message(parsed_error, exc.reason),
                "data": parsed_error,
            }
        except error.URLError as exc:
            return {
                "ok": False,
                "url": redact_url(url),
                "status_code": None,
                "error_type": "network_error",
                "message": f"RapidAPI bodyparts request failed: {exc.reason}",
                "data": None,
            }

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
    def _classify_error(status_code: int) -> str:
        if status_code == 401:
            return "authentication_error"
        if status_code == 403:
            return "permission_denied"
        if status_code == 404:
            return "not_found"
        if status_code == 429:
            return "rate_limit_exceeded"
        return f"http_{status_code}"

    @staticmethod
    def _extract_error_message(parsed_error: Any, fallback_reason: str) -> str:
        if isinstance(parsed_error, dict):
            for key in ("message", "error", "detail", "info"):
                value = parsed_error.get(key)
                if value:
                    return str(value)
        if isinstance(parsed_error, str) and parsed_error.strip():
            return parsed_error
        return str(fallback_reason)



def get_rapidapi_bodyparts_tool():
    return RapidApiBodypartsTool()
