import re
from typing import Any

from app.agent.nodes.base_node import BaseNode
from app.agent.state.session_state import SessionState


INTENT_MAP = {
    "健身": "健身计划",
    "健身计划": "健身计划",
    "饮食": "饮食计划",
    "饮食计划": "饮食计划",
    "调整": "调整计划",
    "调整计划": "调整计划",
    "反馈": "反馈",
    "闲聊": "闲聊",
}


class IntentNode(BaseNode):
    def __call__(self, state: SessionState) -> SessionState:
        user_message = self.latest_user_text(state)
        prompt = f"""
        你是健身 Agent 的意图识别器。
        请识别用户意图，可选范围包括：健身计划、饮食计划、调整计划、反馈、闲聊。
        允许输出多个意图。不要输出范围外的意图。

        同时抽取用户输入中的关键信息：
        - 当日饮食：用户今天吃了什么，若没有则返回空数组
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
            "intent": ["健身计划"],
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

        data = self.invoke_json(prompt)
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
            "available_time_minutes": self._extract_int(
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

        self._merge_memory(state, daily_diet, training_feedback, extracted_profile)

        print("=" * 20)
        print("IntentNode")
        print("=" * 20)
        print(data)

        return state

    @staticmethod
    def _ensure_str_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

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
        parsed = IntentNode._extract_int(user_message, llm_value)
        if parsed and 0 < parsed < 120:
            return parsed
        return None

    @staticmethod
    def _extract_int(user_message: str, llm_value: Any) -> int | None:
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

    @staticmethod
    def _merge_unique_text(target: list[str], values: list[str]) -> list[str]:
        seen = {item.strip().lower() for item in target if item.strip()}
        for value in values:
            normalized = value.strip().lower()
            if normalized and normalized not in seen:
                target.append(value)
                seen.add(normalized)
        return target

    def _merge_memory(
        self,
        state: SessionState,
        daily_diet: list[str],
        training_feedback: list[str],
        profile: dict[str, Any],
    ) -> None:
        long_term = state.memory.long_term_memory
        physical = long_term.physical_profile
        lifestyle = long_term.lifestyle_profile
        dietary = long_term.dietary_profile
        mid_term = state.memory.mid_term_memory

        if profile["name"]:
            long_term.name = profile["name"]
        if profile["job"]:
            long_term.job = profile["job"]
        if profile["gender"]:
            long_term.gender = profile["gender"]

        if profile["height_cm"] is not None:
            physical.height_cm = profile["height_cm"]
        if profile["weight_kg"] is not None:
            physical.weight_kg = profile["weight_kg"]
        if profile["age"] is not None:
            physical.age = profile["age"]
        if profile["body_condition"]:
            physical.body_condition = profile["body_condition"]

        if profile["activity_level"]:
            lifestyle.activity_level = profile["activity_level"]
        if profile["exercise_intensity"]:
            lifestyle.exercise_intensity = profile["exercise_intensity"]
        if profile["available_time_minutes"] is not None:
            lifestyle.available_cooking_time_minutes = profile["available_time_minutes"]
        if profile["goal"]:
            lifestyle.goal = profile["goal"]

        if profile["diet"]:
            dietary.diet = profile["diet"]
        dietary.intolerances = self._merge_unique_text(dietary.intolerances, profile["intolerances"])
        dietary.preferred_ingredients = self._merge_unique_text(
            dietary.preferred_ingredients,
            profile["preferred_ingredients"],
        )
        dietary.disliked_ingredients = self._merge_unique_text(
            dietary.disliked_ingredients,
            profile["disliked_ingredients"],
        )
        dietary.preferred_cuisines = self._merge_unique_text(
            dietary.preferred_cuisines,
            profile["preferred_cuisines"],
        )

        mid_term.daily_diet = self._merge_unique_text(mid_term.daily_diet, daily_diet)
        mid_term.training_feedbacks = self._merge_unique_text(
            mid_term.training_feedbacks,
            training_feedback,
        )
