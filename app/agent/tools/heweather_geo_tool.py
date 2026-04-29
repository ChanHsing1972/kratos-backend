import gzip
import json
from typing import Any
from urllib import error, parse, request

from pydantic import BaseModel, Field

from app.core.config import settings


class HeWeatherGeoLookupInput(BaseModel):
    location: str = Field(
        ...,
        description=(
            "Location keyword to search, such as '苏州', 'Suzhou', 'Beijing', or a district/city name."
        ),
    )
    adm: str | None = Field(
        default=None,
        description="Optional superior administrative region, such as province/state name.",
    )
    range: str | None = Field(
        default="world",
        description="Search scope such as cn or world.",
    )
    lang: str | None = Field(default=None, description="Optional language code for response.")
    number: int | None = Field(default=10, ge=1, le=20, description="Max number of location results.")


class HeWeatherGeoLookupTool:
    name = "heweather_geo_lookup"
    description = (
        "Resolve a natural-language place name into a QWeather LocationID and location metadata. "
        "Use this before calling weather tools when the user provides a city/district name like '苏州'."
    )
    args_schema = HeWeatherGeoLookupInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = HeWeatherGeoLookupInput(**args)

        if not settings.HEWEATHER_API_KEY:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "HEWEATHER_API_KEY is not configured.",
                "url": None,
                "data": None,
            }

        query_params = {
            "location": payload.location,
            "key": settings.HEWEATHER_API_KEY,
        }
        if payload.adm:
            query_params["adm"] = payload.adm
        if payload.range:
            query_params["range"] = payload.range
        if payload.lang:
            query_params["lang"] = payload.lang
        if payload.number:
            query_params["number"] = payload.number

        url = (
            f"{settings.HEWEATHER_API_BASE_URL.rstrip('/')}/geo/v2/city/lookup?"
            f"{parse.urlencode(query_params, doseq=True)}"
        )
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
                    "url": url,
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
                "url": url,
                "location": payload.location,
                "status_code": exc.code,
                "error_type": self._classify_error(exc.code, parsed_error),
                "message": self._extract_error_message(parsed_error, exc.reason),
                "data": parsed_error,
            }
        except error.URLError as exc:
            return {
                "ok": False,
                "url": url,
                "location": payload.location,
                "status_code": None,
                "error_type": "network_error",
                "message": f"HeWeather Geo API request failed: {exc.reason}",
                "data": None,
            }

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



def get_heweather_geo_lookup_tool():
    return HeWeatherGeoLookupTool()
