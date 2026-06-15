"""Nearby place search and navigation advisor.

The tool composes AMap geocoding, nearby POI search, distance, and route APIs
into one stable contract for general "find a nearby place and guide me there"
queries. It is intentionally not sports-specific, so planning can use it for
basketball courts, parks, gyms, cafes, clinics, and other POIs.
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field

from app.agent.tools.amap_location_tool import AMapGeocodeTool, AMapPlaceSearchAroundTool
from app.agent.tools.amap_route_tool import (
    AMapBicyclingRouteTool,
    AMapDistanceTool,
    AMapDrivingRouteTool,
    AMapWalkingRouteTool,
)


TravelMode = Literal["walking", "driving", "bicycling"]


class PlaceNavigationAdvisorInput(BaseModel):
    start_location: str = Field(
        ...,
        description="User starting place, such as 苏州科技城 or 上海人民广场.",
    )
    destination_query: str = Field(
        ...,
        description="Place/POI category to find, such as 室外篮球场, 公园, 健身房, 咖啡店.",
    )
    city: str | None = Field(default=None, description="Optional city name to improve geocoding and POI search.")
    radius_m: int = Field(default=5000, ge=500, le=50000, description="Nearby search radius in meters.")
    max_candidates: int = Field(default=3, ge=1, le=5, description="Maximum nearby places to evaluate.")
    travel_modes: list[TravelMode] = Field(
        default_factory=lambda: ["walking", "driving", "bicycling"],
        description="Route modes to summarize.",
    )


class PlaceNavigationAdvisorTool:
    """Search a nearby destination category and return practical navigation options."""

    name = "place_navigation_advisor"
    description = (
        "Find nearby places by POI keyword/category from a start location, then summarize distance and "
        "walking/driving/bicycling navigation options. Use for general nearby-place and navigation requests."
    )
    args_schema = PlaceNavigationAdvisorInput

    def __init__(self) -> None:
        self.geocode_tool = AMapGeocodeTool()
        self.place_search_tool = AMapPlaceSearchAroundTool()
        self.distance_tool = AMapDistanceTool()
        self.walking_tool = AMapWalkingRouteTool()
        self.driving_tool = AMapDrivingRouteTool()
        self.bicycling_tool = AMapBicyclingRouteTool()

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = PlaceNavigationAdvisorInput(**args)
        request_budget = _RequestBudget(limit=12)

        geocode_result, start = self._resolve_start_location(payload.start_location, payload.city, request_budget)
        if not geocode_result.get("ok"):
            return {
                "ok": False,
                "tool": self.name,
                "step": "geocode",
                "message": geocode_result.get("message") or "起点位置解析失败。",
                "geocode_result": geocode_result,
            }
        if not start:
            return {
                "ok": False,
                "tool": self.name,
                "step": "geocode",
                "message": f"未能识别起点位置：{payload.start_location}。",
                "geocode_result": geocode_result,
            }

        start_coord = start.get("location")
        if not start_coord:
            return {
                "ok": False,
                "tool": self.name,
                "step": "geocode",
                "message": f"起点位置缺少有效坐标：{payload.start_location}。",
                "geocode_result": geocode_result,
            }

        resolved_city = payload.city or self._extract_city_from_geocode(start)
        search_result, pois = self._search_places(
            start_coord=start_coord,
            city=resolved_city,
            destination_query=payload.destination_query,
            radius_m=payload.radius_m,
            max_candidates=payload.max_candidates,
            request_budget=request_budget,
        )
        if not search_result.get("ok"):
            return {
                "ok": False,
                "tool": self.name,
                "step": "place_search_around",
                "message": search_result.get("message") or "附近地点搜索失败。",
                "geocode_result": geocode_result,
                "search_result": search_result,
            }
        if not pois:
            return {
                "ok": False,
                "tool": self.name,
                "step": "place_search_around",
                "message": "附近没有找到匹配地点。可以尝试扩大范围，或换一个更明确的地点类别。",
                "geocode_result": geocode_result,
                "search_result": search_result,
            }

        candidates: list[dict[str, Any]] = []
        for poi in pois[: payload.max_candidates]:
            if not request_budget.allow(1):
                break
            candidate = self._build_candidate(
                start_coord=start_coord,
                poi=poi,
                travel_modes=payload.travel_modes,
                request_budget=request_budget,
            )
            if candidate:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item.get("distance_from_start_m") or 10**9)
        if not candidates:
            return {
                "ok": False,
                "tool": self.name,
                "step": "route_summary",
                "message": "找到了附近地点，但暂时无法生成可用的导航摘要。",
                "geocode_result": geocode_result,
                "search_result": search_result,
            }

        best = candidates[0]
        return {
            "ok": True,
            "tool": self.name,
            "start": {
                "input": payload.start_location,
                "resolved_name": start.get("formatted_address") or start.get("district") or payload.start_location,
                "coordinate": start_coord,
                "city": resolved_city,
            },
            "destination_query": payload.destination_query,
            "radius_m": payload.radius_m,
            "candidates": candidates,
            "best_candidate": {
                "name": best.get("name"),
                "address": best.get("address"),
                "distance_from_start_m": best.get("distance_from_start_m"),
                "routes": best.get("routes"),
                "navigation_url": best.get("navigation_url"),
            },
            "summary": self._build_summary(best, payload.destination_query),
            "request_budget": request_budget.summary(),
            "geocode_result": geocode_result,
            "search_result": search_result,
        }

    def _resolve_start_location(
        self,
        start_location: str,
        city: str | None,
        request_budget: "_RequestBudget",
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        candidates = self._build_location_candidates(start_location, city)
        city_candidates = [city, self._simplify_city(city) if city else None, None]
        deduped_cities = _dedupe([item for item in city_candidates if item or item is None])
        last_result: dict[str, Any] | None = None

        for location_candidate in candidates:
            for city_candidate in deduped_cities:
                if not request_budget.consume():
                    return {"ok": False, "message": "位置解析调用次数已达上限，请提供更明确的起点。"}, None
                result = self.geocode_tool.invoke(
                    {
                        "address": location_candidate,
                        "city": city_candidate,
                        "output": "JSON",
                    }
                )
                last_result = result
                if not result.get("ok"):
                    continue
                geocodes = ((result.get("data") or {}).get("geocodes") or [])
                if geocodes:
                    return result, geocodes[0]

        return last_result or {"ok": False, "message": "起点位置解析失败。"}, None

    def _search_places(
        self,
        *,
        start_coord: str,
        city: str | None,
        destination_query: str,
        radius_m: int,
        max_candidates: int,
        request_budget: "_RequestBudget",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        keyword_candidates = self._build_keyword_candidates(destination_query)
        radius_candidates = _dedupe([radius_m, min(max(radius_m + 3000, radius_m), 50000)])
        last_result: dict[str, Any] | None = None

        for radius in radius_candidates:
            for keywords in keyword_candidates:
                if not request_budget.consume():
                    return last_result or {"ok": False, "message": "附近地点搜索调用次数已达上限。"}, []
                result = self.place_search_tool.invoke(
                    {
                        "location": start_coord,
                        "keywords": keywords,
                        "radius": radius,
                        "offset": max_candidates,
                        "city": city,
                        "sortrule": "distance",
                        "output": "JSON",
                    }
                )
                last_result = result
                if not result.get("ok"):
                    continue
                pois = ((result.get("data") or {}).get("pois") or [])
                filtered = [poi for poi in pois if isinstance(poi, dict) and poi.get("location")]
                if filtered:
                    return result, filtered

        return last_result or {"ok": False, "message": "附近地点搜索失败。"}, []

    def _build_candidate(
        self,
        *,
        start_coord: str,
        poi: dict[str, Any],
        travel_modes: list[TravelMode],
        request_budget: "_RequestBudget",
    ) -> dict[str, Any] | None:
        poi_location = str(poi.get("location") or "")
        if not poi_location:
            return None

        distance_m = None
        if request_budget.consume():
            distance_result = self.distance_tool.invoke(
                {
                    "origins": start_coord,
                    "destination": poi_location,
                    "type": "3",
                    "output": "JSON",
                }
            )
            distance_m = self._extract_distance_m(distance_result)

        routes: dict[str, dict[str, Any]] = {}
        for mode in _dedupe(travel_modes):
            if not request_budget.consume():
                break
            route_result = self._invoke_route_tool(mode, start_coord, poi_location)
            summary = self._extract_route_summary(route_result)
            if summary:
                routes[mode] = summary

        name = str(poi.get("name") or "目的地")
        address = _format_poi_address(poi)
        return {
            "name": name,
            "address": address,
            "location": poi_location,
            "type": poi.get("type"),
            "tel": poi.get("tel"),
            "distance_from_start_m": distance_m,
            "routes": routes,
            "navigation_url": self._navigation_url(poi_location, name),
        }

    def _invoke_route_tool(self, mode: TravelMode, origin: str, destination: str) -> dict[str, Any]:
        if mode == "walking":
            return self.walking_tool.invoke({"origin": origin, "destination": destination, "output": "JSON"})
        if mode == "driving":
            return self.driving_tool.invoke(
                {
                    "origin": origin,
                    "destination": destination,
                    "extensions": "base",
                    "output": "JSON",
                }
            )
        return self.bicycling_tool.invoke({"origin": origin, "destination": destination})

    @staticmethod
    def _extract_distance_m(distance_result: dict[str, Any]) -> int | None:
        if not distance_result.get("ok"):
            return None
        results = ((distance_result.get("data") or {}).get("results") or [])
        if not results:
            return None
        try:
            return int(float(results[0].get("distance")))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_route_summary(route_result: dict[str, Any]) -> dict[str, Any] | None:
        if not route_result.get("ok"):
            return None
        data = route_result.get("data") or {}
        route = data.get("route") if isinstance(data, dict) else {}
        paths = []
        if isinstance(route, dict):
            paths = route.get("paths") or []
        if not paths and isinstance(data, dict):
            paths = (data.get("data") or {}).get("paths") if isinstance(data.get("data"), dict) else data.get("paths") or []
        if not paths:
            return None
        path = paths[0]
        if not isinstance(path, dict):
            return None
        steps = path.get("steps") or []
        instructions = [
            str(step.get("instruction"))
            for step in steps[:5]
            if isinstance(step, dict) and step.get("instruction")
        ]
        return {
            "distance_m": _int_or_none(path.get("distance")),
            "duration_s": _int_or_none(path.get("duration")),
            "instructions_preview": instructions,
        }

    @staticmethod
    def _build_location_candidates(start_location: str, city: str | None) -> list[str]:
        cleaned = start_location.strip()
        for old, new in [("我在", ""), ("附近", ""), ("周边", ""), ("这边", ""), ("这里", "")]:
            cleaned = cleaned.replace(old, new)
        cleaned = cleaned.strip(" ，。,.？?")
        candidates = [start_location.strip(), cleaned]
        if city and cleaned and city not in cleaned:
            candidates.append(f"{city}{cleaned}")
        return _dedupe([item for item in candidates if item])

    @staticmethod
    def _build_keyword_candidates(destination_query: str) -> list[str]:
        cleaned = destination_query.strip(" ，。,.？?")
        candidates = [cleaned]
        replacements = [
            ("室外篮球场", "篮球场"),
            ("户外篮球场", "篮球场"),
            ("篮球场地", "篮球场"),
            ("室外球场", "篮球场 体育场"),
            ("运动场地", "体育场 运动场"),
        ]
        for needle, replacement in replacements:
            if needle in cleaned:
                candidates.append(replacement)
        if "篮球" in cleaned:
            candidates.extend(["篮球场", "体育公园 篮球场", "运动场 篮球"])
        candidates.append(cleaned.replace("一个", "").replace("一家", "").strip())
        return _dedupe([item for item in candidates if item])

    @staticmethod
    def _extract_city_from_geocode(geocode: dict[str, Any]) -> str | None:
        city = geocode.get("city")
        if isinstance(city, list):
            return city[0] if city else None
        if isinstance(city, str) and city:
            return city
        district = geocode.get("district")
        if isinstance(district, str) and district:
            return district
        return None

    @staticmethod
    def _simplify_city(city: str | None) -> str | None:
        if not city:
            return None
        return city.replace("我在", "").replace("附近", "").replace("周边", "").strip() or None

    @staticmethod
    def _navigation_url(location: str, name: str) -> str:
        encoded_name = quote(name)
        return (
            "https://uri.amap.com/navigation"
            f"?to={location},{encoded_name}&mode=car&policy=1&src=kratos&coordinate=gaode&callnative=0"
        )

    @staticmethod
    def _build_summary(best: dict[str, Any], destination_query: str) -> dict[str, Any]:
        return {
            "recommended_place": best.get("name"),
            "destination_query": destination_query,
            "address": best.get("address"),
            "distance_from_start_m": best.get("distance_from_start_m"),
            "available_route_modes": list((best.get("routes") or {}).keys()),
        }


class _RequestBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def allow(self, cost: int = 1) -> bool:
        return self.used + cost <= self.limit

    def consume(self, cost: int = 1) -> bool:
        if not self.allow(cost):
            return False
        self.used += cost
        return True

    def summary(self) -> dict[str, int]:
        return {"used": self.used, "limit": self.limit}


def _format_poi_address(poi: dict[str, Any]) -> str:
    address = poi.get("address")
    if isinstance(address, list):
        address = " ".join(str(item) for item in address if item)
    if address:
        return str(address)
    return " ".join(str(poi.get(key) or "") for key in ["pname", "cityname", "adname"]).strip()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _dedupe(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def get_place_navigation_advisor_tool() -> PlaceNavigationAdvisorTool:
    return PlaceNavigationAdvisorTool()
