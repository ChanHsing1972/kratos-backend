"""QWeather 实时天气和天气预报工具。"""

import gzip
import json
from typing import Any, Literal
from urllib import error, parse, request

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.agent.tools.http_utils import redact_url


class HeWeatherInput(BaseModel):
    endpoint: Literal["weather_now", "weather_forecast"] = Field(
        default="weather_now",
        description=(
            "Which QWeather endpoint to call. Use 'weather_now' for current weather, "
            "or 'weather_forecast' for daily forecast."
        ),
    )
    location: str = Field(
        ...,
        description=(
            "Required location parameter. Use a QWeather LocationID like '101010100' or a longitude,latitude pair "
            "like '116.41,39.92'."
        ),
    )
    days: Literal["3d", "7d", "10d", "15d", "30d"] | None = Field(
        default="3d",
        description="Required when endpoint='weather_forecast'. Supported values: 3d, 7d, 10d, 15d, 30d.",
    )
    lang: str | None = Field(default=None, description="Optional language code for QWeather response.")
    unit: Literal["m", "i"] | None = Field(
        default="m",
        description="Unit system. 'm' for metric, 'i' for imperial.",
    )

    @model_validator(mode="after")
    def validate_shape(self):
        if self.endpoint == "weather_now":
            return self
        if self.endpoint == "weather_forecast" and not self.days:
            raise ValueError("days is required when endpoint='weather_forecast'")
        return self


class HeWeatherTool:
    name = "heweather_weather"
    description = (
        "Get real-time or daily forecast weather from QWeather/HeWeather using API Key authentication. "
        "Supports /v7/weather/now and /v7/weather/{days}. Input must include a LocationID or longitude,latitude."
    )
    args_schema = HeWeatherInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = HeWeatherInput(**args)

        if not settings.HEWEATHER_API_KEY:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "HEWEATHER_API_KEY is not configured.",
                "url": None,
                "data": None,
            }

        path = self._build_path(payload)
        query_params = self._build_query_params(payload)
        url = f"{settings.HEWEATHER_API_BASE_URL.rstrip('/')}{path}?{parse.urlencode(query_params, doseq=True)}"

        req = request.Request(
            url=url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": "Kratos-Agent/1.0",
            },
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=settings.HEWEATHER_TIMEOUT_SECONDS) as response:
                raw_body = self._decode_response_body(response.read(), response.headers.get("Content-Encoding"))
                data = self._load_json_or_text(raw_body)
                return {
                    "ok": self._is_success(data),
                    "url": redact_url(url),
                    "endpoint": payload.endpoint,
                    "days": payload.days,
                    "location": payload.location,
                    "status_code": response.status,
                    "headers": dict(response.headers.items()),
                    "data": data,
                }
        except error.HTTPError as exc:
            error_body = self._decode_response_body(exc.read(), exc.headers.get("Content-Encoding"))
            parsed_error = self._load_json_or_text(error_body)
            return {
                "ok": False,
                "url": redact_url(url),
                "endpoint": payload.endpoint,
                "days": payload.days,
                "location": payload.location,
                "status_code": exc.code,
                "error_type": self._classify_error(exc.code, parsed_error),
                "message": self._extract_error_message(parsed_error, exc.reason),
                "data": parsed_error,
            }
        except error.URLError as exc:
            return {
                "ok": False,
                "url": redact_url(url),
                "endpoint": payload.endpoint,
                "days": payload.days,
                "location": payload.location,
                "status_code": None,
                "error_type": "network_error",
                "message": f"HeWeather API request failed: {exc.reason}",
                "data": None,
            }

    @staticmethod
    def _build_path(payload: HeWeatherInput) -> str:
        if payload.endpoint == "weather_now":
            return "/v7/weather/now"
        return f"/v7/weather/{payload.days}"

    @staticmethod
    def _build_query_params(payload: HeWeatherInput) -> dict[str, Any]:
        params: dict[str, Any] = {
            "location": payload.location,
            "key": settings.HEWEATHER_API_KEY,
        }
        if payload.lang:
            params["lang"] = payload.lang
        if payload.unit:
            params["unit"] = payload.unit
        return params

    @staticmethod
    def _decode_response_body(body: bytes, content_encoding: str | None) -> str:
        raw = body
        if content_encoding and "gzip" in content_encoding.lower():
            raw = gzip.decompress(body)
        return raw.decode("utf-8", errors="ignore")

    @staticmethod
    def _load_json_or_text(raw_body: str) -> Any:
        if not raw_body:
            return None
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body

    @staticmethod
    def _is_success(data: Any) -> bool:
        return isinstance(data, dict) and str(data.get("code")) == "200"

    @staticmethod
    def _classify_error(status_code: int, parsed_error: Any) -> str:
        if status_code == 401:
            return "authentication_error"
        if status_code == 402:
            return "quota_or_plan_error"
        if status_code == 404:
            return "not_found"
        if status_code == 429:
            return "rate_limit_exceeded"
        if isinstance(parsed_error, dict):
            code = str(parsed_error.get("code") or "")
            if code and code != "200":
                return f"qweather_{code}"
        return f"http_{status_code}"

    @staticmethod
    def _extract_error_message(parsed_error: Any, fallback_reason: str) -> str:
        if isinstance(parsed_error, dict):
            nested_error = parsed_error.get("error")
            if isinstance(nested_error, dict):
                for key in ("title", "detail", "type", "status"):
                    value = nested_error.get(key)
                    if value:
                        return str(value)
            for key in ("message", "msg", "error", "code"):
                value = parsed_error.get(key)
                if value:
                    return str(value)
        if isinstance(parsed_error, str) and parsed_error.strip():
            return parsed_error
        return str(fallback_reason)



def get_heweather_tool():
    return HeWeatherTool()
