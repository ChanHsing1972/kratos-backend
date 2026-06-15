"""Intent policy helpers for Agent planning, tools, and reflection."""

from __future__ import annotations

import re
from typing import Any

from app.agent.state.reasoning import Task
from app.agent.state.session_state import SessionState
from app.agent.tool_planner import repair_tool_args


PLAN_INTENTS = {"健身计划", "饮食计划", "调整计划"}
REFLECTION_INTENTS = PLAN_INTENTS | {"风险评估"}
INFO_INTENTS = {"信息查询", "天气查询", "新闻搜索", "路线查询", "普通问答"}


def has_plan_intent(intents: list[str]) -> bool:
    return any(intent in PLAN_INTENTS for intent in intents)


def should_run_reflection(state: SessionState) -> bool:
    """Return whether the final answer needs the training/diet quality gate."""

    if any(intent in REFLECTION_INTENTS for intent in state.reasoning.intent):
        return True
    if state.result.workout_plan is not None or state.result.diet_plan is not None:
        return True
    user_message = latest_user_text(state)
    return is_pain_or_safety_request(user_message)


def is_weather_query(text: str) -> bool:
    return "天气" in text or "气温" in text or "下雨" in text or "适合运动" in text or "适合跑步" in text


def is_news_query(text: str) -> bool:
    if any(keyword in text for keyword in ["新闻", "资讯", "报道", "动态"]):
        return True
    if "链接" in text and any(keyword in text for keyword in ["给出", "搜索", "最近", "最新"]):
        return True
    return "最新" in text and any(keyword in text for keyword in ["领域", "行业", "有哪些"])


def is_route_query(text: str) -> bool:
    if is_running_route_query(text):
        return True
    if any(keyword in text for keyword in ["导航", "怎么去", "怎么走", "路线规划", "规划路线", "路径", "到达", "距离"]):
        return True
    if any(keyword in text for keyword in ["步行", "骑行", "驾车", "开车", "打车", "公交", "地铁"]):
        return any(place_word in text for place_word in ["去", "到", "附近", "周边", "路线"])
    place_words = "篮球场|球场|足球场|网球场|羽毛球馆|游泳馆|公园|健身房|场馆|场地|门店|医院|餐厅|咖啡|停车场"
    return bool(
        re.search(rf"(附近|周边).{{0,24}}?(找|搜索|推荐|有没有|哪里有|{place_words})", text)
        or re.search(rf"(找|搜索|推荐|查找).{{0,24}}?(附近|周边|{place_words})", text)
    )


def is_running_route_query(text: str) -> bool:
    return any(keyword in text for keyword in ["跑步路线", "晨跑路线", "夜跑路线", "公里路线", "适合跑步", "跑步"])


def is_pain_or_safety_request(text: str) -> bool:
    return any(keyword in text for keyword in ["疼", "疼痛", "不舒服", "受伤", "扭伤", "膝盖", "腰", "肩", "风险"])


def is_explicit_training_plan_request(text: str) -> bool:
    explicit_generation_keywords = [
        "生成训练计划",
        "制定训练计划",
        "训练计划",
        "健身计划",
        "安排训练",
        "今日训练",
        "本次训练",
    ]
    if any(keyword in text for keyword in explicit_generation_keywords):
        return True
    if is_news_query(text) or is_weather_query(text):
        return False
    return any(
        keyword in text
        for keyword in [
            "练腿",
            "练胸",
            "练背",
            "上肢训练",
            "下肢训练",
        ]
    )


def information_tool_calls(
    state: SessionState,
    task: Task,
    user_message: str,
    available_tools: list[str],
) -> list[dict[str, Any]]:
    """Choose tools for information-query tasks without leaking plan tools."""

    available = set(available_tools)
    task_text = f"{task.name} {task.description or ''}"
    calls: list[dict[str, Any]] = []
    wants_weather = is_weather_query(task_text) or ("天气查询" in state.reasoning.intent and "天气" in task.name)
    wants_news = is_news_query(task_text) or ("新闻搜索" in state.reasoning.intent and "新闻" in task.name)
    wants_route = is_route_query(task_text) or ("路线查询" in state.reasoning.intent and "路线" in task.name)

    if wants_weather and "weather_fitness_advisor" in available:
        calls.append(
            {
                "tool_name": "weather_fitness_advisor",
                "args": repair_tool_args("weather_fitness_advisor", {}, user_message, task, state),
                "id": "intent-weather",
            }
        )

    if wants_news and "tavily_search" in available:
        calls.append(
            {
                "tool_name": "tavily_search",
                "args": {"query": _search_query(user_message)},
                "id": "intent-news-search",
            }
        )

    if wants_route and not is_running_route_query(user_message) and "place_navigation_advisor" in available:
        calls.append(
            {
                "tool_name": "place_navigation_advisor",
                "args": repair_tool_args("place_navigation_advisor", {}, user_message, task, state),
                "id": "intent-place-navigation",
            }
        )
    elif wants_route and "running_route_advisor" in available:
        calls.append(
            {
                "tool_name": "running_route_advisor",
                "args": repair_tool_args("running_route_advisor", {}, user_message, task, state),
                "id": "intent-running-route",
            }
        )

    return calls


def summarize_tool_result(result: Any) -> str:
    """Compress tool output into a short observation for task results."""

    if not isinstance(result, dict):
        return str(result)

    tool_name = result.get("tool") or result.get("name") or ""
    if tool_name == "weather_fitness_advisor":
        if result.get("ok") is False:
            return f"天气查询失败：{result.get('message') or '未能获取天气'}。"
        summary = result.get("weather_summary") or {}
        advice = result.get("fitness_advice") or []
        weather = (
            f"{summary.get('label') or ''}{summary.get('city') or ''}天气："
            f"{summary.get('day_text') or '未知'}，"
            f"{summary.get('temp_min_c') or '?'}-{summary.get('temp_max_c') or '?'}℃。"
        )
        if advice:
            return f"{weather}运动建议：{'；'.join(str(item) for item in advice[:2])}"
        return weather

    if tool_name == "place_navigation_advisor":
        if result.get("ok") is False:
            return f"地点与导航查询失败：{result.get('message') or '未能获取地点和路线'}。"
        start = result.get("start") or {}
        best = result.get("best_candidate") or {}
        candidates = result.get("candidates") or []
        candidate_lines = []
        for item in candidates[:3]:
            if not isinstance(item, dict):
                continue
            distance = item.get("distance_from_start_m")
            distance_text = f"，距起点约 {distance} 米" if distance is not None else ""
            routes = item.get("routes") or {}
            route_modes = "、".join(routes.keys()) if isinstance(routes, dict) else ""
            route_text = f"，可用路线：{route_modes}" if route_modes else ""
            candidate_lines.append(f"{item.get('name') or '地点'}（{item.get('address') or '地址未返回'}{distance_text}{route_text}）")
        summary = (
            f"从{start.get('resolved_name') or start.get('input') or '起点'}附近搜索"
            f"{result.get('destination_query') or '目的地'}，推荐 {best.get('name') or '候选地点'}。"
        )
        if candidate_lines:
            return summary + "候选：" + "；".join(candidate_lines)
        return summary

    if "results" in result and isinstance(result["results"], list):
        items = []
        for item in result["results"][:3]:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or "搜索结果"
            url = item.get("url") or item.get("link")
            items.append(f"{title} {url}".strip())
        return "搜索返回结果：" + "；".join(items) if items else "搜索返回结果为空。"

    if "answer" in result:
        return str(result["answer"])
    if "message" in result:
        return str(result["message"])
    return str(result)[:800]


def latest_user_text(state: SessionState) -> str:
    for message in reversed(state.conversation.messages):
        if getattr(message, "type", None) == "human":
            content = getattr(message, "content", "")
            if isinstance(content, list):
                return "\n".join(
                    str(item.get("text") or "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            return str(content)
    return ""


def _search_query(user_message: str) -> str:
    text = user_message.strip()
    segments = [segment.strip() for segment in re.split(r"[。！？!?；;\n]", text) if segment.strip()]
    news_segments = [segment for segment in segments if is_news_query(segment)]
    query_text = " ".join(news_segments) if news_segments else text
    if "新闻" in query_text or "最新" in query_text or "资讯" in query_text:
        return query_text
    return f"{query_text} 最新资讯 链接"
