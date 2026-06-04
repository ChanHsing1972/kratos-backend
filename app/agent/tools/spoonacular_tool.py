"""Spoonacular 食谱搜索工具。"""

import json
from typing import Any
from urllib import error, parse, request

from pydantic import BaseModel, Field

from app.core.config import settings
from app.agent.tools.http_utils import redact_url


class SpoonacularRecipeSearchInput(BaseModel):
    query_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Query parameters for GET /recipes/complexSearch. Example: "
            "{'query': 'pasta', 'maxFat': 25, 'number': 2, 'diet': 'vegetarian'}."
        ),
    )


class SpoonacularRecipeSearchTool:
    name = "spoonacular_recipe_search"
    description = (
        "Search recipes using Spoonacular complexSearch with filters such as query, cuisine, diet, intolerances, "
        "ingredients, meal type, ready time, servings, sorting, and nutrition constraints. "
        "Use this tool for recipe discovery, 'what's in the fridge' style matching, and filtered meal suggestions."
    )
    args_schema = SpoonacularRecipeSearchInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = SpoonacularRecipeSearchInput(**args)

        if not settings.SPOONACULAR_API_KEY:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "SPOONACULAR_API_KEY is not configured.",
                "url": None,
                "data": None,
            }

        params = dict(payload.query_params)
        params["apiKey"] = settings.SPOONACULAR_API_KEY
        query_string = parse.urlencode(self._normalize_params(params), doseq=True)
        url = f"{settings.SPOONACULAR_API_BASE_URL.rstrip('/')}/recipes/complexSearch?{query_string}"

        req = request.Request(
            url=url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Kratos-Agent/1.0",
            },
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=settings.SPOONACULAR_TIMEOUT_SECONDS) as response:
                raw_body = response.read().decode("utf-8")
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
                "message": f"Spoonacular API request failed: {exc.reason}",
                "data": None,
            }

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, bool):
                normalized[key] = str(value).lower()
            else:
                normalized[key] = value
        return normalized

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
        if status_code == 402:
            return "quota_or_plan_error"
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
            for key in ("message", "detail", "status"):
                value = parsed_error.get(key)
                if value:
                    return str(value)
        if isinstance(parsed_error, str) and parsed_error.strip():
            return parsed_error
        return str(fallback_reason)



def get_spoonacular_recipe_search_tool():
    return SpoonacularRecipeSearchTool()
