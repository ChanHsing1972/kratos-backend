from typing import Any


def build_validation_fallback(tool_name: str, args: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool_name,
        "fallback": True,
        "validation_failed": True,
        "reason": f"参数校验失败：{error}",
        "args": args,
        "suggestions": _fallback_suggestions(tool_name),
    }


def build_error_fallback(tool_name: str, args: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool_name,
        "fallback": True,
        "validation_failed": False,
        "reason": str(error),
        "args": args,
        "suggestions": _fallback_suggestions(tool_name),
    }


def _fallback_suggestions(tool_name: str) -> list[str]:
    if "weather" in tool_name or "heweather" in tool_name:
        return ["无法获取实时天气时，优先选择室内训练。", "如需户外运动，降低强度并避开极端温度、降雨和大风。"]
    if "route" in tool_name or "amap" in tool_name:
        return ["无法获取地图路线时，优先选择熟悉、照明良好、可中途返回的路线。", "按目标距离的一半折返，避免陌生路段。"]
    if "diet" in tool_name or "spoonacular" in tool_name:
        return ["无法联网查食谱时，按蛋白质、主食、蔬菜三类组合餐盘。", "先记录饮食限制，再给出保守食材建议。"]
    if "musclewiki" in tool_name or "rapidapi" in tool_name:
        return ["外部动作库不可用时，优先使用本地动作库或低风险替代动作。"]
    return ["工具不可用时，基于已知信息给出保守建议，并明确不确定性。"]
