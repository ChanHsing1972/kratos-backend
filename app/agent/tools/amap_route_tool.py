"""高德距离和路线规划工具。"""

import json
from typing import Any, Literal
from urllib import error, parse, request

from pydantic import BaseModel, Field

from app.core.config import settings
from app.agent.tools.http_utils import redact_url


class AMapDistanceInput(BaseModel):
    origins: str = Field(
        ...,
        description=(
            "One or more origin coordinates. Format: 'lon,lat' or multiple points joined by '|', "
            "for example '116.481028,39.989643|114.481028,39.989643'."
        ),
    )
    destination: str = Field(..., description="Destination coordinate in 'lon,lat' format.")
    type: Literal["0", "1", "3"] | None = Field(
        default="1",
        description="Distance mode: 0 straight-line, 1 driving, 3 walking.",
    )
    output: Literal["JSON", "XML"] | None = Field(default="JSON")


class AMapWalkingRouteInput(BaseModel):
    origin: str = Field(..., description="Origin coordinate in 'lon,lat' format.")
    destination: str = Field(..., description="Destination coordinate in 'lon,lat' format.")
    origin_id: str | None = Field(default=None, description="Optional origin POI ID.")
    destination_id: str | None = Field(default=None, description="Optional destination POI ID.")
    output: Literal["JSON", "XML"] | None = Field(default="JSON")


class AMapTransitRouteInput(BaseModel):
    origin: str = Field(..., description="Origin coordinate in 'lon,lat' format.")
    destination: str = Field(..., description="Destination coordinate in 'lon,lat' format.")
    city: str = Field(..., description="Origin city name or citycode.")
    cityd: str | None = Field(default=None, description="Optional destination city for intercity transit.")
    extensions: Literal["base", "all"] | None = Field(default="base")
    strategy: Literal["0", "1", "2", "3", "5"] | None = Field(default="0")
    nightflag: Literal["0", "1"] | None = Field(default="0")
    date: str | None = Field(default=None, description="Optional departure date, for example 2026-04-29.")
    time: str | None = Field(default=None, description="Optional departure time, for example 22:34.")
    output: Literal["JSON", "XML"] | None = Field(default="JSON")


class AMapDrivingRouteInput(BaseModel):
    origin: str = Field(..., description="Origin coordinate in 'lon,lat' format.")
    destination: str = Field(..., description="Destination coordinate in 'lon,lat' format.")
    strategy: str | None = Field(default="10", description="Driving strategy code.")
    waypoints: str | None = Field(default=None, description="Optional waypoint coordinates joined by ';'.")
    avoidpolygons: str | None = Field(default=None, description="Optional polygons to avoid.")
    province: str | None = Field(default=None, description="Province abbreviation for plate restriction checks.")
    number: str | None = Field(default=None, description="Plate number without province prefix.")
    cartype: Literal["0", "1", "2"] | None = Field(default="0")
    ferry: Literal["0", "1"] | None = Field(default="0")
    roadaggregation: Literal["true", "false"] | None = Field(default="false")
    nosteps: Literal["0", "1"] | None = Field(default="0")
    extensions: Literal["base", "all"] | None = Field(default="base")
    output: Literal["JSON", "XML"] | None = Field(default="JSON")


class AMapBicyclingRouteInput(BaseModel):
    origin: str = Field(..., description="Origin coordinate in 'lon,lat' format.")
    destination: str = Field(..., description="Destination coordinate in 'lon,lat' format.")


class _AMapBaseTool:
    @staticmethod
    def _ensure_api_key() -> dict[str, Any] | None:
        if settings.AMAP_WEB_API_KEY:
            return None
        return {
            "ok": False,
            "status_code": None,
            "error_type": "configuration_error",
            "message": "AMAP_WEB_API_KEY is not configured.",
            "url": None,
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
            for key in ("info", "errmsg", "errdetail", "message", "detail", "infocode", "errcode"):
                value = parsed_error.get(key)
                if value:
                    return str(value)
        if isinstance(parsed_error, str) and parsed_error.strip():
            return parsed_error
        return str(fallback_reason)

    @staticmethod
    def _request(path: str, params: dict[str, Any]) -> dict[str, Any]:
        query_string = parse.urlencode(_AMapBaseTool._normalize_params(params), doseq=True)
        url = f"{settings.AMAP_WEB_API_BASE_URL.rstrip('/')}{path}?{query_string}"
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
                data = _AMapBaseTool._load_json_or_text(raw_body)
                return {
                    "ok": _AMapBaseTool._is_success(data),
                    "url": redact_url(url),
                    "status_code": response.status,
                    "headers": dict(response.headers.items()),
                    "data": data,
                    "message": (
                        None
                        if _AMapBaseTool._is_success(data)
                        else _AMapBaseTool._extract_error_message(data, "AMap request failed.")
                    ),
                }
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            parsed_error = _AMapBaseTool._load_json_or_text(error_body)
            return {
                "ok": False,
                "url": redact_url(url),
                "status_code": exc.code,
                "error_type": _AMapBaseTool._classify_error(exc.code),
                "message": _AMapBaseTool._extract_error_message(parsed_error, exc.reason),
                "data": parsed_error,
            }
        except error.URLError as exc:
            return {
                "ok": False,
                "url": redact_url(url),
                "status_code": None,
                "error_type": "network_error",
                "message": f"AMap API request failed: {exc.reason}",
                "data": None,
            }

    @staticmethod
    def _is_success(data: Any) -> bool:
        if isinstance(data, dict):
            if str(data.get("status")) == "1":
                return True
            if str(data.get("errcode")) == "0":
                return True
        return False


class AMapDistanceTool(_AMapBaseTool):
    name = "amap_distance"
    description = "Measure straight-line, driving, or walking distance using AMap distance API."
    args_schema = AMapDistanceInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapDistanceInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "origins": payload.origins,
            "destination": payload.destination,
            "type": payload.type,
            "output": payload.output,
        }
        return self._request("/v3/distance", params)


class AMapWalkingRouteTool(_AMapBaseTool):
    name = "amap_walking_route"
    description = "Plan a walking route using AMap walking direction API."
    args_schema = AMapWalkingRouteInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapWalkingRouteInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "origin": payload.origin,
            "destination": payload.destination,
            "origin_id": payload.origin_id,
            "destination_id": payload.destination_id,
            "output": payload.output,
        }
        return self._request("/v3/direction/walking", params)


class AMapTransitRouteTool(_AMapBaseTool):
    name = "amap_transit_route"
    description = "Plan a public transit route using AMap integrated transit direction API."
    args_schema = AMapTransitRouteInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapTransitRouteInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "origin": payload.origin,
            "destination": payload.destination,
            "city": payload.city,
            "cityd": payload.cityd,
            "extensions": payload.extensions,
            "strategy": payload.strategy,
            "nightflag": payload.nightflag,
            "date": payload.date,
            "time": payload.time,
            "output": payload.output,
        }
        return self._request("/v3/direction/transit/integrated", params)


class AMapDrivingRouteTool(_AMapBaseTool):
    name = "amap_driving_route"
    description = "Plan a driving route using AMap driving direction API."
    args_schema = AMapDrivingRouteInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapDrivingRouteInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "origin": payload.origin,
            "destination": payload.destination,
            "strategy": payload.strategy,
            "waypoints": payload.waypoints,
            "avoidpolygons": payload.avoidpolygons,
            "province": payload.province,
            "number": payload.number,
            "cartype": payload.cartype,
            "ferry": payload.ferry,
            "roadaggregation": payload.roadaggregation,
            "nosteps": payload.nosteps,
            "extensions": payload.extensions,
            "output": payload.output,
        }
        return self._request("/v3/direction/driving", params)


class AMapBicyclingRouteTool(_AMapBaseTool):
    name = "amap_bicycling_route"
    description = "Plan a bicycling route using AMap bicycling direction API."
    args_schema = AMapBicyclingRouteInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapBicyclingRouteInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "origin": payload.origin,
            "destination": payload.destination,
        }
        return self._request("/v4/direction/bicycling", params)



def get_amap_distance_tool():
    return AMapDistanceTool()



def get_amap_walking_route_tool():
    return AMapWalkingRouteTool()



def get_amap_transit_route_tool():
    return AMapTransitRouteTool()



def get_amap_driving_route_tool():
    return AMapDrivingRouteTool()



def get_amap_bicycling_route_tool():
    return AMapBicyclingRouteTool()
