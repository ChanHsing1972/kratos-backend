from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.tools.heweather_geo_tool import HeWeatherGeoLookupTool
from app.agent.tools.heweather_tool import HeWeatherTool


class WeatherFitnessAdvisorInput(BaseModel):
    city: str = Field(..., description="City, district, or place name such as 苏州 or Beijing.")
    when: Literal["today", "tomorrow"] = Field(
        default="tomorrow",
        description="Whether to summarize today's or tomorrow's weather for fitness guidance.",
    )
    lang: str | None = Field(default=None, description="Optional language code for QWeather response.")
    unit: Literal["m", "i"] | None = Field(default="m", description="Unit system.")
    include_fitness_advice: bool = Field(
        default=True,
        description="Whether to include fitness advice based on the weather.",
    )


class WeatherFitnessAdvisorTool:
    name = "weather_fitness_advisor"
    description = (
        "Resolve a natural-language city name, fetch weather forecast automatically, summarize today's or tomorrow's weather, "
        "and optionally provide fitness advice suitable for the weather. Use this when the user asks about weather in a city "
        "or asks whether the weather is suitable for exercise."
    )
    args_schema = WeatherFitnessAdvisorInput

    def __init__(self):
        self.geo_tool = HeWeatherGeoLookupTool()
        self.weather_tool = HeWeatherTool()

    def invoke(self, args: dict[str, Any]) -> dict[str, Any]:
        payload = WeatherFitnessAdvisorInput(**args)

        geo_result = self.geo_tool.invoke(
            {
                "location": payload.city,
                "range": "cn",
                "lang": payload.lang,
                "number": 5,
            }
        )
        if not geo_result.get("ok"):
            return {
                "ok": False,
                "tool": self.name,
                "step": "geo_lookup",
                "message": geo_result.get("message") or "Failed to resolve city.",
                "geo_result": geo_result,
            }

        locations = ((geo_result.get("data") or {}).get("location") or [])
        if not locations:
            return {
                "ok": False,
                "tool": self.name,
                "step": "geo_lookup",
                "message": f"未找到城市“{payload.city}”对应的位置结果。",
                "geo_result": geo_result,
            }

        best_location = locations[0]
        location_id = best_location.get("id")
        if not location_id:
            return {
                "ok": False,
                "tool": self.name,
                "step": "geo_lookup",
                "message": f"城市“{payload.city}”的位置结果缺少 LocationID。",
                "geo_result": geo_result,
            }

        weather_result = self.weather_tool.invoke(
            {
                "endpoint": "weather_forecast",
                "location": location_id,
                "days": "3d",
                "lang": payload.lang,
                "unit": payload.unit,
            }
        )
        if not weather_result.get("ok"):
            return {
                "ok": False,
                "tool": self.name,
                "step": "weather_forecast",
                "message": weather_result.get("message") or "Failed to fetch weather forecast.",
                "geo_result": geo_result,
                "weather_result": weather_result,
            }

        daily = ((weather_result.get("data") or {}).get("daily") or [])
        target_index = 0 if payload.when == "today" else 1
        if len(daily) <= target_index:
            return {
                "ok": False,
                "tool": self.name,
                "step": "weather_forecast",
                "message": f"天气预报结果中缺少 {payload.when} 对应的数据。",
                "geo_result": geo_result,
                "weather_result": weather_result,
            }

        target_day = daily[target_index]
        weather_summary = self._build_weather_summary(best_location, target_day, payload.when)
        fitness_advice = self._build_fitness_advice(target_day) if payload.include_fitness_advice else []

        return {
            "ok": True,
            "tool": self.name,
            "city": payload.city,
            "when": payload.when,
            "resolved_location": {
                "id": best_location.get("id"),
                "name": best_location.get("name"),
                "adm1": best_location.get("adm1"),
                "adm2": best_location.get("adm2"),
                "country": best_location.get("country"),
                "lat": best_location.get("lat"),
                "lon": best_location.get("lon"),
            },
            "weather_summary": weather_summary,
            "fitness_advice": fitness_advice,
            "geo_result": geo_result,
            "weather_result": weather_result,
        }

    @staticmethod
    def _build_weather_summary(location: dict[str, Any], target_day: dict[str, Any], when: str) -> dict[str, Any]:
        label = "今天" if when == "today" else "明天"
        return {
            "label": label,
            "city": location.get("name"),
            "province": location.get("adm1"),
            "date": target_day.get("fxDate"),
            "day_text": target_day.get("textDay"),
            "night_text": target_day.get("textNight"),
            "temp_min_c": target_day.get("tempMin"),
            "temp_max_c": target_day.get("tempMax"),
            "wind_dir_day": target_day.get("windDirDay"),
            "wind_scale_day": target_day.get("windScaleDay"),
            "wind_speed_day_kmh": target_day.get("windSpeedDay"),
            "humidity_percent": target_day.get("humidity"),
            "precip_mm": target_day.get("precip"),
            "uv_index": target_day.get("uvIndex"),
            "pressure_hpa": target_day.get("pressure"),
            "visibility_km": target_day.get("vis"),
        }

    @staticmethod
    def _build_fitness_advice(target_day: dict[str, Any]) -> list[str]:
        tips: list[str] = []

        text_day = str(target_day.get("textDay") or "")
        precip = WeatherFitnessAdvisorTool._to_float(target_day.get("precip")) or 0.0
        temp_max = WeatherFitnessAdvisorTool._to_float(target_day.get("tempMax"))
        temp_min = WeatherFitnessAdvisorTool._to_float(target_day.get("tempMin"))
        uv_index = WeatherFitnessAdvisorTool._to_float(target_day.get("uvIndex"))
        humidity = WeatherFitnessAdvisorTool._to_float(target_day.get("humidity"))
        wind_speed = WeatherFitnessAdvisorTool._to_float(target_day.get("windSpeedDay"))

        rainy_keywords = ["雨", "雪", "雷"]
        if any(keyword in text_day for keyword in rainy_keywords) or precip >= 5:
            tips.append("明天有明显降水，更建议选择室内训练，如力量训练、划船机、动感单车或自重循环。")
            tips.append("如果必须外出运动，建议缩短时长并做好防滑、防雨和保暖准备。")
        else:
            tips.append("如果你计划慢跑、快走或骑行，明天整体具备一定户外训练条件。")

        if temp_max is not None and temp_max <= 10:
            tips.append("气温偏低，运动前建议延长热身时间，重点活动膝踝和髋部。")
        elif temp_max is not None and temp_max >= 30:
            tips.append("气温较高，建议避免正午户外高强度训练，并注意补水与电解质补充。")

        if temp_min is not None and temp_max is not None and (temp_max - temp_min) >= 8:
            tips.append("昼夜温差较大，外出训练建议采用便于增减的分层穿衣。")

        if humidity is not None and humidity >= 80:
            tips.append("湿度较高，体感会更闷，建议适当降低配速或训练密度。")

        if wind_speed is not None and wind_speed >= 25:
            tips.append("白天风速偏大，户外跑步或球类训练时要注意风阻和保暖。")

        if uv_index is not None and uv_index >= 8:
            tips.append("紫外线较强，若安排白天户外训练，建议做好防晒并尽量避开中午时段。")

        if not tips:
            tips.append("明天天气对一般训练影响不大，可以按原计划进行，并注意常规补水和热身。")

        return tips

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None



def get_weather_fitness_advisor_tool():
    return WeatherFitnessAdvisorTool()
