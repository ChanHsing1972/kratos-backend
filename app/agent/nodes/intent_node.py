"""意图识别节点。

本节点只负责识别本轮用户目标并抽取“本轮可用信息”。抽取出的健康数据不会
直接写入长期记忆，而是放入 `reasoning.extracted_info` 和 `memory.ephemeral_turn_info`。
需要入库的数据由服务层生成 `pending_confirmation_updates`，等待用户确认。
"""

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agent.intent_policy import (
    is_news_query,
    is_route_query,
    is_weather_query,
    is_explicit_training_plan_request,
)
from app.agent.nodes.base_node import BaseNode
from app.agent.state.session_state import SessionState


INTENT_MAP = {
    "健身": "健身",
    "健身计划": "健身计划",
    "饮食": "饮食",
    "饮食计划": "饮食计划",
    "饮食记录": "饮食记录",
    "记录饮食": "饮食记录",
    "饮食打卡": "饮食记录",
    "调整": "调整",
    "调整计划": "调整计划",
    "反馈": "反馈",
    "闲聊": "闲聊",
    "信息查询": "信息查询",
    "天气": "天气查询",
    "天气查询": "天气查询",
    "新闻": "新闻搜索",
    "新闻搜索": "新闻搜索",
    "搜索": "新闻搜索",
    "路线": "路线查询",
    "路线查询": "路线查询",
    "普通问答": "普通问答",
}


class IntentExtraction(BaseModel):
    """Structured contract for one-turn intent and slot extraction."""

    intent: list[str] = Field(default_factory=lambda: ["闲聊"])
    daily_diet: list[str] = Field(default_factory=list)
    training_feedback: list[str] = Field(default_factory=list)
    name: str | None = None
    job: str | None = None
    gender: str | None = None
    age: int | float | str | None = None
    height_cm: int | float | str | None = None
    weight_kg: int | float | str | None = None
    body_condition: str | None = None
    goal: str | None = None
    activity_level: str | None = None
    exercise_intensity: str | None = None
    available_time_minutes: int | float | str | None = None
    diet: str | None = None
    intolerances: list[str] = Field(default_factory=list)
    preferred_ingredients: list[str] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)
    preferred_cuisines: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0, le=1)


class IntentNode(BaseNode):
    """识别用户意图并抽取本轮 profile、饮食和训练反馈信息。"""

    def __call__(self, state: SessionState) -> SessionState:
        """更新 `state.reasoning.intent` 与 `state.reasoning.extracted_info`。"""

        user_message = self.latest_user_text(state)
        data = self._deterministic_intent_data(user_message)
        if data is None:
            skill_context = self.describe_active_skills(state)
            conversation_context = json.dumps(
                self.recent_conversation_context(state),
                ensure_ascii=False,
                default=str,
                indent=2,
            )
            conversation_summaries = json.dumps(
                state.conversation.summaries[-3:],
                ensure_ascii=False,
                default=str,
                indent=2,
            )
            prompt = f"""
            你是健身 Agent 的意图识别器。
            请识别用户意图，可选范围包括：健身计划、饮食计划、饮食记录、调整计划、反馈、闲聊、信息查询、天气查询、新闻搜索、路线查询、普通问答。
            允许输出多个意图。不要输出范围外的意图。
            天气、新闻、搜索链接、路线查询属于信息查询，不要归类为健身计划，除非用户明确要求生成训练安排。
            如果启用了 Skill，请结合 Skill 的适用场景辅助判断用户目标，但不要把 Skill 名称当作意图。

            启用 Skill:
            {skill_context}

            最近会话上下文(JSON):
            {conversation_context}

            会话摘要(JSON):
            {conversation_summaries}

            上下文理解规则：
            - 用户可能用“继续”“这个”“上一个”“第二个”“说错了”“不是”“改成”等省略或修正表达。
            - 必须先结合最近会话上下文判断真实指代，再识别意图和抽取信息。
            - 如果上下文仍无法唯一确定指代，不要猜测，按普通问答或反馈处理，并让后续回复澄清。

            同时抽取用户输入中的关键信息：
            - 当日饮食：用户今天吃了什么；如果用户只是想开始记录但没有提供食物，返回空数组
            - 训练反馈：用户对训练的感受、疲劳、疼痛、完成情况等，若没有则返回空数组
            - 用户姓名：若提到名字则提取，否则返回 null
            - 职业/身份：如上班族、学生、程序员、教师等，若提到则返回，否则返回 null
            - 性别：若能确定则返回，否则返回 null
            - 年龄：若提到则返回整数，否则返回 null
            - 身高厘米：若提到则返回数值，否则返回 null
            - 体重公斤：若提到则返回数值，否则返回 null
            - 身体状况：如过瘦、偏胖、膝盖不适、久坐、睡眠差等，没有则返回 null
            - 健身/饮食目标：如增肌、减脂、保持、塑形，没有则返回 null
            - 活动水平：如低、中、高，没有则返回 null
            - 训练强度：如低、中、高，没有则返回 null
            - 可用做饭/训练时间（分钟）：若提到则返回整数，否则返回 null
            - 饮食方式：如 vegetarian、vegan、高蛋白、低脂等，没有则返回 null
            - 过敏/不耐受：返回数组，没有则返回空数组
            - 喜欢的食材：返回数组，没有则返回空数组
            - 不喜欢的食材：返回数组，没有则返回空数组
            - 偏好菜系：返回数组，没有则返回空数组

            用户输入:
            {user_message}

            严格输出一个 JSON 对象，不要 Markdown：
            {{
                "intent": ["信息查询"],
                "daily_diet": ["鸡胸肉", "米饭"],
                "training_feedback": ["今天腿很酸", "跑步完成了 30 分钟"],
                "name": "Will",
                "job": "上班族",
                "gender": null,
                "age": null,
                "height_cm": 180,
                "weight_kg": 75,
                "body_condition": null,
                "goal": "增肌",
                "activity_level": null,
                "exercise_intensity": null,
                "available_time_minutes": 30,
                "diet": null,
                "intolerances": [],
                "preferred_ingredients": [],
                "disliked_ingredients": [],
                "preferred_cuisines": []
            }}
            """

            data = self.invoke_json(prompt, state)
        data = self._validate_intent_payload(data)
        intents = data.get("intent") or ["闲聊"]
        if isinstance(intents, str):
            intents = [intents]

        normalized_intents: list[str] = []
        for intent in intents:
            normalized = INTENT_MAP.get(str(intent).strip())
            if normalized and normalized not in normalized_intents:
                normalized_intents.append(normalized)
        if not normalized_intents:
            normalized_intents = ["闲聊"]

        daily_diet = self._ensure_str_list(data.get("daily_diet") or [])
        training_feedback = self._ensure_str_list(data.get("training_feedback") or [])
        intolerances = self._ensure_str_list(data.get("intolerances") or [])
        preferred_ingredients = self._ensure_str_list(data.get("preferred_ingredients") or [])
        disliked_ingredients = self._ensure_str_list(data.get("disliked_ingredients") or [])
        preferred_cuisines = self._ensure_str_list(data.get("preferred_cuisines") or [])

        extracted_profile = {
            "name": self._extract_name(user_message, data.get("name")),
            "job": self._extract_job(user_message, data.get("job")),
            "gender": self._clean_str(data.get("gender")),
            "age": self._extract_age(user_message, data.get("age")),
            "height_cm": self._extract_height_cm(user_message, data.get("height_cm")),
            "weight_kg": self._extract_weight_kg(user_message, data.get("weight_kg")),
            "body_condition": self._clean_str(data.get("body_condition")),
            "goal": self._clean_str(data.get("goal")),
            "activity_level": self._clean_str(data.get("activity_level")),
            "exercise_intensity": self._clean_str(data.get("exercise_intensity")),
            "available_time_minutes": self._extract_available_time_minutes(
                user_message,
                data.get("available_time_minutes"),
            ),
            "diet": self._clean_str(data.get("diet")),
            "intolerances": intolerances,
            "preferred_ingredients": preferred_ingredients,
            "disliked_ingredients": disliked_ingredients,
            "preferred_cuisines": preferred_cuisines,
        }

        state.reasoning.intent = normalized_intents
        state.reasoning.extracted_info = {
            "daily_diet": daily_diet,
            "training_feedback": training_feedback,
            "profile": extracted_profile,
        }
        state.memory.ephemeral_turn_info = state.reasoning.extracted_info

        self.logger.debug(
            "IntentNode result: %s",
            {
                "raw": data,
                "intent": state.reasoning.intent,
                "extracted_info": state.reasoning.extracted_info,
            },
        )

        return state

    @staticmethod
    def _ensure_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _deterministic_intent_data(user_message: str) -> dict[str, Any] | None:
        """Fast path for common app modes to avoid an LLM round trip."""

        text = user_message.strip()
        if not text:
            return None

        intents: list[str] = []
        if re.search(r"(工具|tools?|可调用|能力清单|支持哪些)", text, flags=re.IGNORECASE):
            intents.append("信息查询")
        if is_explicit_training_plan_request(text):
            intents.append("健身计划")
        diet_record = IntentNode._is_diet_record_request(text)
        if diet_record:
            intents.append("饮食记录")
        if any(keyword in text for keyword in ["饮食计划", "食谱", "吃什么", "怎么吃", "餐单"]):
            intents.append("饮食计划")
        if any(keyword in text for keyword in ["调整计划", "调整训练", "调整一下", "太累", "疼", "不舒服", "受伤", "疲劳"]):
            intents.append("调整计划" if "健身计划" in intents else "反馈")
        if not intents and any(keyword in text for keyword in ["完成", "反馈", "打卡", "感受"]):
            intents.append("反馈")
        if is_weather_query(text):
            intents.append("天气查询")
        if is_news_query(text):
            intents.append("新闻搜索")
        if is_route_query(text):
            intents.append("路线查询")
        if any(intent in intents for intent in ["天气查询", "新闻搜索", "路线查询"]) and "信息查询" not in intents:
            intents.append("信息查询")

        if not intents:
            return None

        goal = None
        for keyword in ["减脂塑形", "减脂", "塑形", "增肌", "保持", "恢复", "提升体能"]:
            if keyword in text:
                goal = keyword
                break

        training_feedback = []
        if any(keyword in text for keyword in ["疼", "不舒服", "受伤", "疲劳", "酸"]):
            training_feedback.append(text[:120])

        daily_diet = IntentNode._extract_daily_diet_text(text) if diet_record else []

        return {
            "intent": intents,
            "daily_diet": daily_diet,
            "training_feedback": training_feedback,
            "name": None,
            "job": None,
            "gender": "男" if "男" in text else ("女" if "女" in text else None),
            "age": IntentNode._extract_age(text, None),
            "height_cm": IntentNode._extract_height_cm(text, None),
            "weight_kg": IntentNode._extract_weight_kg(text, None),
            "body_condition": text[:120] if any(keyword in text for keyword in ["疼", "不舒服", "受伤", "久坐", "睡眠差"]) else None,
            "goal": goal,
            "activity_level": None,
            "exercise_intensity": None,
            "available_time_minutes": IntentNode._extract_available_time_minutes(text, None),
            "diet": None,
            "intolerances": [],
            "preferred_ingredients": [],
            "disliked_ingredients": [],
            "preferred_cuisines": [],
            "confidence": 0.75,
        }

    @staticmethod
    def _validate_intent_payload(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return IntentExtraction().model_dump()
        try:
            return IntentExtraction.model_validate(data).model_dump()
        except ValidationError:
            return IntentExtraction(
                intent=IntentNode._ensure_str_list(data.get("intent") or ["闲聊"]),
                daily_diet=IntentNode._ensure_str_list(data.get("daily_diet") or []),
                training_feedback=IntentNode._ensure_str_list(data.get("training_feedback") or []),
            ).model_dump()

    @staticmethod
    def _is_diet_record_request(text: str) -> bool:
        record_markers = [
            "记录饮食",
            "保存饮食",
            "饮食打卡",
            "记一餐",
            "记一下",
            "帮我记",
            "我吃了",
            "今天吃了",
            "刚吃了",
            "早餐",
            "午餐",
            "晚餐",
            "夜宵",
            "加餐",
        ]
        if any(marker in text for marker in record_markers):
            return "吃什么" not in text and "怎么吃" not in text
        return False

    @staticmethod
    def _extract_daily_diet_text(text: str) -> list[str]:
        patterns = [
            r"(?:我|今天|刚刚|早餐|午餐|晚餐|夜宵|加餐)?\s*(?:吃了|喝了|摄入了)\s*([^。！？!?；;\n]{1,120})",
            r"(?:早餐|午餐|晚餐|夜宵|加餐)\s*[:：是为]?\s*([^。！？!?；;\n]{1,120})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip(" ：:,，。")
                return [value] if value else []
        return []

    @staticmethod
    def _clean_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _extract_name(user_message: str, llm_name: Any) -> str | None:
        if llm_name:
            text = str(llm_name).strip()
            if text:
                return text
        patterns = [
            r"我叫\s*([A-Za-z][A-Za-z0-9_-]{0,30})",
            r"我的名字是\s*([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,30})",
            r"叫我\s*([A-Za-z\u4e00-\u9fff][A-Za-z0-9_\-\u4e00-\u9fff]{0,30})",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_job(user_message: str, llm_job: Any) -> str | None:
        if llm_job:
            text = str(llm_job).strip()
            if text:
                return text
        job_keywords = ["上班族", "学生", "程序员", "教师", "老师", "医生", "护士", "司机", "自由职业"]
        for keyword in job_keywords:
            if keyword in user_message:
                return keyword
        match = re.search(r"我是(?:一位|一个|个)?([\u4e00-\u9fffA-Za-z]{2,12})(?:，|。|,|\s|$)", user_message)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate not in {"男生", "女生", "男人", "女人"}:
                return candidate
        return None

    @staticmethod
    def _extract_height_cm(user_message: str, llm_value: Any) -> float | None:
        parsed = IntentNode._to_float(llm_value)
        if parsed is not None:
            if 0 < parsed <= 3:
                return round(parsed * 100, 1)
            return parsed

        match_m = re.search(r"身高(?:是|为)?\s*(\d+(?:\.\d+)?)\s*米", user_message)
        if match_m:
            return round(float(match_m.group(1)) * 100, 1)
        match_cm = re.search(r"身高(?:是|为)?\s*(\d+(?:\.\d+)?)\s*(?:厘米|cm|CM)", user_message)
        if match_cm:
            return float(match_cm.group(1))
        return None

    @staticmethod
    def _extract_weight_kg(user_message: str, llm_value: Any) -> float | None:
        parsed = IntentNode._to_float(llm_value)
        if parsed is not None:
            return parsed
        match = re.search(r"体重(?:是|为)?\s*(\d+(?:\.\d+)?)\s*(?:kg|KG|公斤|千克)", user_message)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _extract_age(user_message: str, llm_value: Any) -> int | None:
        parsed = IntentNode._extract_age_value(user_message, llm_value)
        if parsed and 0 < parsed < 120:
            return parsed
        return None

    @staticmethod
    def _extract_age_value(user_message: str, llm_value: Any) -> int | None:
        if isinstance(llm_value, int):
            return llm_value
        if isinstance(llm_value, float):
            return int(llm_value)
        if isinstance(llm_value, str):
            text = llm_value.strip()
            if text.isdigit():
                return int(text)
        match = re.search(r"(?:年龄|我|今年)?\s*(\d{1,3})\s*岁", user_message)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_available_time_minutes(user_message: str, llm_value: Any) -> int | None:
        if isinstance(llm_value, int):
            return llm_value
        if isinstance(llm_value, float):
            return int(llm_value)
        if isinstance(llm_value, str):
            text = llm_value.strip()
            if text.isdigit():
                return int(text)
        match = re.search(r"(\d+)\s*分钟", user_message)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None
