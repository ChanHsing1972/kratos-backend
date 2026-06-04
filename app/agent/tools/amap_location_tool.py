"""高德地理编码和周边 POI 搜索工具。"""

from typing import Any

from pydantic import BaseModel, Field

from app.agent.tools.amap_route_tool import _AMapBaseTool
from app.core.config import settings


class AMapGeocodeInput(BaseModel):
    address: str = Field(..., description="Address or place name to geocode.")
    city: str | None = Field(default=None, description="Optional city name or citycode to improve geocoding accuracy.")
    output: str | None = Field(default="JSON", description="Output format, recommended JSON.")


class AMapPlaceSearchAroundInput(BaseModel):
    location: str = Field(..., description="Center coordinate in 'lon,lat' format.")
    keywords: str | None = Field(default=None, description="Optional search keywords such as 公园, 绿道, 操场.")
    types: str | None = Field(default=None, description="Optional AMap POI type codes.")
    radius: int | None = Field(default=3000, ge=1, le=50000, description="Search radius in meters.")
    sortrule: str | None = Field(default="distance", description="Sort rule: distance or weight.")
    page: int | None = Field(default=1, ge=1)
    offset: int | None = Field(default=10, ge=1, le=25)
    city: str | None = Field(default=None, description="Optional city restriction.")
    output: str | None = Field(default="JSON", description="Output format, recommended JSON.")


class AMapGeocodeTool(_AMapBaseTool):
    name = "amap_geocode"
    description = "Convert a natural-language address or place name into AMap coordinates and geocode metadata."
    args_schema = AMapGeocodeInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapGeocodeInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "address": payload.address,
            "city": payload.city,
            "output": payload.output,
        }
        return self._request("/v3/geocode/geo", params)


class AMapPlaceSearchAroundTool(_AMapBaseTool):
    name = "amap_place_search_around"
    description = (
        "Search nearby POIs around a coordinate using AMap place around API. "
        "Useful for finding nearby parks, greenways, tracks, and sports-friendly places."
    )
    args_schema = AMapPlaceSearchAroundInput

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = AMapPlaceSearchAroundInput(**args)
        config_error = self._ensure_api_key()
        if config_error:
            return config_error

        params = {
            "key": settings.AMAP_WEB_API_KEY,
            "location": payload.location,
            "keywords": payload.keywords,
            "types": payload.types,
            "radius": payload.radius,
            "sortrule": payload.sortrule,
            "page": payload.page,
            "offset": payload.offset,
            "city": payload.city,
            "output": payload.output,
        }
        return self._request("/v3/place/around", params)



def get_amap_geocode_tool():
    return AMapGeocodeTool()



def get_amap_place_search_around_tool():
    return AMapPlaceSearchAroundTool()
