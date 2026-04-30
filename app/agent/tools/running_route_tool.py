from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.tools.amap_location_tool import AMapGeocodeTool, AMapPlaceSearchAroundTool
from app.agent.tools.amap_route_tool import AMapDistanceTool, AMapWalkingRouteTool


class RunningRouteAdvisorInput(BaseModel):
    start_location: str = Field(..., description="User starting place, such as 苏州工业园区金鸡湖 or 人民广场.")
    city: str | None = Field(default=None, description="Optional city name to improve geocoding and nearby search.")
    target_distance_km: float = Field(default=5.0, ge=1.0, le=30.0, description="Desired running distance in kilometers.")
    route_preference: Literal["park_loop", "greenway", "track", "general"] = Field(
        default="park_loop",
        description="Preferred running environment.",
    )
    radius_m: int = Field(default=5000, ge=500, le=10000, description="Nearby search radius in meters.")
    max_candidates: int = Field(default=5, ge=1, le=10, description="Maximum nearby candidate places to evaluate.")


class RunningRouteAdvisorTool:
    name = "running_route_advisor"
    description = (
        "Recommend nearby running routes by geocoding the user's start location, searching nearby suitable running places, "
        "estimating route distance, and returning practical running suggestions."
    )
    args_schema = RunningRouteAdvisorInput

    def __init__(self):
        self.geocode_tool = AMapGeocodeTool()
        self.place_search_tool = AMapPlaceSearchAroundTool()
        self.distance_tool = AMapDistanceTool()
        self.walking_tool = AMapWalkingRouteTool()

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = RunningRouteAdvisorInput(**args)

        geocode_result, start = self._resolve_start_location(payload.start_location, payload.city)
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

        search_result, pois = self._search_running_places(
            start_coord=start_coord,
            city=payload.city or self._extract_city_from_geocode(start),
            route_preference=payload.route_preference,
            radius_m=payload.radius_m,
            max_candidates=payload.max_candidates,
        )
        if not search_result.get("ok"):
            return {
                "ok": False,
                "tool": self.name,
                "step": "place_search_around",
                "message": search_result.get("message") or "附近适跑地点搜索失败。",
                "geocode_result": geocode_result,
                "search_result": search_result,
            }

        if not pois:
            return {
                "ok": False,
                "tool": self.name,
                "step": "place_search_around",
                "message": "附近没有找到合适的跑步地点。可以尝试扩大搜索范围或换一个更明确的起点描述。",
                "geocode_result": geocode_result,
                "search_result": search_result,
            }

        recommendations: list[dict[str, Any]] = []
        for poi in pois[: payload.max_candidates]:
            candidate = self._build_candidate(start_coord, poi, payload.target_distance_km)
            if candidate is None:
                continue
            recommendations.append(candidate)

        recommendations.sort(key=lambda item: item["score"], reverse=True)
        top_routes = recommendations[:3]

        if not top_routes:
            return {
                "ok": False,
                "tool": self.name,
                "step": "route_scoring",
                "message": "找到了附近地点，但暂时无法生成合适的跑步路线建议。",
                "geocode_result": geocode_result,
                "search_result": search_result,
            }

        summary = self._build_summary(top_routes[0], payload.target_distance_km)

        return {
            "ok": True,
            "tool": self.name,
            "start": {
                "input": payload.start_location,
                "resolved_name": start.get("formatted_address") or start.get("district") or payload.start_location,
                "coordinate": start_coord,
                "city": self._extract_city_from_geocode(start) or payload.city,
            },
            "target_distance_km": payload.target_distance_km,
            "route_preference": payload.route_preference,
            "recommended_routes": top_routes,
            "summary": summary,
            "running_tips": self._build_running_tips(payload.target_distance_km, top_routes[0]),
            "geocode_result": geocode_result,
            "search_result": search_result,
        }

    def _resolve_start_location(self, start_location: str, city: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
        candidates = self._build_location_candidates(start_location)
        last_result: dict[str, Any] | None = None

        city_candidates: list[str | None] = []
        if city:
            city_candidates.append(city)
            city_candidates.append(self._simplify_city(city))
        city_candidates.append(None)

        deduped_city_candidates: list[str | None] = []
        for item in city_candidates:
            if item not in deduped_city_candidates:
                deduped_city_candidates.append(item)

        for location_candidate in candidates:
            for city_candidate in deduped_city_candidates:
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

    def _search_running_places(
        self,
        start_coord: str,
        city: str | None,
        route_preference: str,
        radius_m: int,
        max_candidates: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        keyword_candidates = self._build_keyword_candidates(route_preference)
        radius_candidates = [radius_m, min(max(radius_m + 2000, radius_m), 10000)]
        last_result: dict[str, Any] | None = None

        for radius in radius_candidates:
            for keywords in keyword_candidates:
                result = self.place_search_tool.invoke(
                    {
                        "location": start_coord,
                        "keywords": keywords,
                        "radius": radius,
                        "offset": max_candidates,
                        "city": city,
                        "output": "JSON",
                    }
                )
                last_result = result
                if not result.get("ok"):
                    continue
                pois = ((result.get("data") or {}).get("pois") or [])
                filtered_pois = [poi for poi in pois if poi.get("location")]
                if filtered_pois:
                    return result, filtered_pois

        return last_result or {"ok": False, "message": "附近地点搜索失败。"}, []


    def _build_candidate(self, start_coord: str, poi: dict[str, Any], target_distance_km: float) -> dict[str, Any] | None:
        poi_location = poi.get("location")
        if not poi_location:
            return None

        distance_result = self.distance_tool.invoke(
            {
                "origins": start_coord,
                "destination": poi_location,
                "type": "3",
                "output": "JSON",
            }
        )
        distance_m = self._extract_distance_m(distance_result)
        if distance_m is None:
            return None

        walking_result = self.walking_tool.invoke(
            {
                "origin": start_coord,
                "destination": poi_location,
                "output": "JSON",
            }
        )
        walking_summary = self._extract_walking_summary(walking_result)

        one_way_km = distance_m / 1000
        round_trip_km = round(one_way_km * 2, 2)
        loop_like_km = round(max(round_trip_km, target_distance_km * 0.8), 2)
        score = self._score_candidate(target_distance_km, round_trip_km, poi)

        return {
            "name": poi.get("name"),
            "address": poi.get("address") or poi.get("pname") or "",
            "location": poi_location,
            "type": poi.get("type"),
            "distance_from_start_m": distance_m,
            "estimated_one_way_km": round(one_way_km, 2),
            "estimated_round_trip_km": round_trip_km,
            "suggested_route_distance_km": loop_like_km,
            "walking_route": walking_summary,
            "reason": self._build_reason(target_distance_km, round_trip_km, poi),
            "score": score,
        }

    @staticmethod
    def _build_keywords(route_preference: str) -> str:
        mapping = {
            "park_loop": "公园 绿道 跑步 健身步道",
            "greenway": "绿道 江边步道 湖边步道 跑步",
            "track": "操场 体育场 田径场",
            "general": "公园 绿道 操场 跑步",
        }
        return mapping.get(route_preference, mapping["general"])

    def _build_keyword_candidates(self, route_preference: str) -> list[str]:
        primary = self._build_keywords(route_preference)
        candidates = [
            primary,
            "公园 绿道 湖边步道 江边步道",
            "公园 绿地 广场 体育场",
            "跑步 健身步道 操场",
            "公园",
        ]
        deduped: list[str] = []
        for item in candidates:
            if item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _build_location_candidates(start_location: str) -> list[str]:
        candidates = [start_location.strip()]
        cleaned = start_location.strip()
        for old, new in [("我在", ""), ("附近", ""), ("周边", ""), ("这边", ""), ("这里", "")]:
            cleaned = cleaned.replace(old, new)
        cleaned = cleaned.strip(" ，。,.？?")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

        if "金鸡湖" in cleaned and "苏州" not in cleaned:
            candidates.append(f"苏州{cleaned}")
        if "工业园区" in cleaned and "苏州" not in cleaned:
            candidates.append(f"苏州{cleaned}")

        deduped: list[str] = []
        for item in candidates:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    @staticmethod
    def _simplify_city(city: str) -> str:
        return city.replace("我在", "").replace("附近", "").replace("周边", "").strip()

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
    def _extract_distance_m(distance_result: dict[str, Any]) -> int | None:
        if not distance_result.get("ok"):
            return None
        results = ((distance_result.get("data") or {}).get("results") or [])
        if not results:
            return None
        distance = results[0].get("distance")
        try:
            return int(float(distance))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_walking_summary(walking_result: dict[str, Any]) -> dict[str, Any] | None:
        if not walking_result.get("ok"):
            return None
        route = ((walking_result.get("data") or {}).get("route") or {})
        paths = route.get("paths") or []
        if not paths:
            return None
        path = paths[0]
        steps = path.get("steps") or []
        instructions = [step.get("instruction") for step in steps[:5] if isinstance(step, dict) and step.get("instruction")]
        return {
            "distance_m": path.get("distance"),
            "duration_s": path.get("duration"),
            "instructions_preview": instructions,
        }

    @staticmethod
    def _score_candidate(target_distance_km: float, round_trip_km: float, poi: dict[str, Any]) -> float:
        diff = abs(round_trip_km - target_distance_km)
        base_score = max(0.0, 100.0 - diff * 18)
        name_text = f"{poi.get('name') or ''} {poi.get('type') or ''}"
        bonus = 0.0
        for keyword in ["公园", "绿道", "步道", "体育", "操场", "湖", "江", "河"]:
            if keyword in name_text:
                bonus += 5.0
        return round(base_score + min(bonus, 20.0), 2)

    @staticmethod
    def _build_reason(target_distance_km: float, round_trip_km: float, poi: dict[str, Any]) -> str:
        reasons = []
        if abs(round_trip_km - target_distance_km) <= 1.5:
            reasons.append("与目标跑量较接近")
        name_text = f"{poi.get('name') or ''} {poi.get('type') or ''}"
        for keyword, label in [("公园", "环境相对舒适"), ("绿道", "更适合连续慢跑"), ("操场", "更适合控配速训练"), ("体育", "运动氛围较好")]:
            if keyword in name_text:
                reasons.append(label)
        if not reasons:
            reasons.append("作为附近可达地点，适合先进行基础跑步尝试")
        return "，".join(reasons[:3])

    @staticmethod
    def _build_summary(best_route: dict[str, Any], target_distance_km: float) -> dict[str, Any]:
        return {
            "recommended_place": best_route.get("name"),
            "target_distance_km": target_distance_km,
            "suggested_route_distance_km": best_route.get("suggested_route_distance_km"),
            "distance_from_start_m": best_route.get("distance_from_start_m"),
            "reason": best_route.get("reason"),
        }

    @staticmethod
    def _build_running_tips(target_distance_km: float, best_route: dict[str, Any]) -> list[str]:
        tips = [
            "建议出发前先热身 8-10 分钟，重点活动踝关节、膝关节和髋部。",
            "初次跑这条路线时可以先用轻松配速熟悉环境，再决定是否加量。",
        ]
        route_km = best_route.get("suggested_route_distance_km")
        if isinstance(route_km, (int, float)) and route_km < target_distance_km:
            tips.append("如果路线略短，可以通过往返补足里程，或在安全区域内多绕 1-2 圈。")
        if target_distance_km >= 8:
            tips.append("目标距离较长，建议补充饮水并尽量选择补给方便、路况熟悉的路线。")
        return tips



def get_running_route_advisor_tool():
    return RunningRouteAdvisorTool()
