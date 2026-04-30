import json
from typing import Any
from urllib import error, parse, request

from pydantic import BaseModel, Field

from app.core.config import settings


class AMapIpLocationInput(BaseModel):
    ip: str | None = Field(
        default=None,
        description=(
            "Optional IPv4 address for geolocation. If omitted, AMap will infer the client IP from the HTTP request. "
            "Only mainland China IP addresses are supported by this endpoint."
        ),
    )
    output: str | None = Field(
        default="JSON",
        description="Optional output format. Recommended value: JSON.",
    )


class AMapIpLocationTool:
    name = "amap_ip_location"
    description = (
        "Locate an IP address using AMap Web Service IP API. Returns province, city, adcode, and bounding rectangle. "
        "Use this for city-level IP geolocation in mainland China."
    )
    args_schema = AMapIpLocationInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapIpLocationInput(**args)

        if not settings.AMAP_WEB_API_KEY:
            return {
                "ok": False,
                "status_code": None,
                "error_type": "configuration_error",
                "message": "AMAP_WEB_API_KEY is not configured.",
                "url": None,
                "data": None,
            }

        params: dict[str, Any] = {
            "key": settings.AMAP_WEB_API_KEY,
        }
        if payload.ip:
            params["ip"] = payload.ip
        if payload.output:
            params["output"] = payload.output

        query_string = parse.urlencode(self._normalize_params(params), doseq=True)
        url = f"{settings.AMAP_WEB_API_BASE_URL.rstrip('/')}/v3/ip?{query_string}"

        req = request.Request(
            url=url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Kratos-Agent/1.0",
            },
            method="GET",
        )

        try:
            with request.urlopen(req, timeout=settings.AMAP_WEB_API_TIMEOUT_SECONDS) as response:
                raw_body = response.read().decode("utf-8", errors="ignore")
                data = self._load_json_or_text(raw_body)
                ok = self._is_success(data)
                inferred_from_request_ip = payload.ip is None
                note = None
                if ok and inferred_from_request_ip:
                    note = (
                        "No explicit IP was provided. This result reflects the outbound IP location of the current runtime/request environment, "
                        "which may not be the actual end-user location."
                    )
                return {
                    "ok": ok,
                    "url": url,
                    "status_code": response.status,
                    "headers": dict(response.headers.items()),
                    "data": data,
                    "requested_ip": payload.ip,
                    "inferred_from_request_ip": inferred_from_request_ip,
                    "note": note,
                    "message": None if ok else self._extract_error_message(data, "AMap IP lookup failed."),
                }
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            parsed_error = self._load_json_or_text(error_body)
            return {
                "ok": False,
                "url": url,
                "status_code": exc.code,
                "error_type": self._classify_error(exc.code),
                "message": self._extract_error_message(parsed_error, exc.reason),
                "data": parsed_error,
            }
        except error.URLError as exc:
            return {
                "ok": False,
                "url": url,
                "status_code": None,
                "error_type": "network_error",
                "message": f"AMap IP API request failed: {exc.reason}",
                "data": None,
            }

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
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
    def _is_success(data: Any) -> bool:
        return isinstance(data, dict) and str(data.get("status")) == "1" and str(data.get("infocode")) == "10000"

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
            for key in ("info", "message", "detail", "infocode"):
                value = parsed_error.get(key)
                if value:
                    return str(value)
        if isinstance(parsed_error, str) and parsed_error.strip():
            return parsed_error
        return str(fallback_reason)



def get_amap_ip_location_tool():
    return AMapIpLocationTool()
