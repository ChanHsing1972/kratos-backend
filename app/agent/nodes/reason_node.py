import json
import re

from app.agent.json_utils import LLMJsonParseError
from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.tools import ToolCall


class ReasonNode(BaseNode):
    def __call__(self, state):
        task = state.reasoning.current_task()
        if task is None:
            return state

        memory_answer = self._answer_from_memory(state)
        if memory_answer is not None:
            task.status = TaskStatus.done
            task.result = memory_answer
            state.reasoning.advance_task()

            print("=" * 20)
            print("ReasonNode - Memory Result")
            print("=" * 20)
            print({"result": memory_answer})

            return state

        convs = state.conversation.conversations
        available_tools = sorted(state.tools.available_tools.keys())
        tool_descriptions = self.describe_tools(state.tools.available_tools)
        user_message = self.latest_user_text(state)
        first_ai_message = state.conversation.first_ai_message or state.result.first_response or ""
        extracted_info = state.reasoning.extracted_info or {}
        memory_context = self._memory_context(state)
        memory_context_json = json.dumps(memory_context, ensure_ascii=False, default=str, indent=2)
        tool_descriptions_json = json.dumps(tool_descriptions, ensure_ascii=False, default=str, indent=2)
        conversations_json = json.dumps(
            [item.model_dump() for item in convs],
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        extracted_info_json = json.dumps(extracted_info, ensure_ascii=False, default=str, indent=2)

        task.status = TaskStatus.running

        prompt_reason = f"""        
        用户原始问题: {user_message}
        第一轮 AI 分析: {first_ai_message}
        当前识别意图: {state.reasoning.intent}
        已提取关键信息(JSON):
        {extracted_info_json}
        你正在执行任务: {task.name}
        任务描述: {task.description}
        
        查询工具之前，先查看过往会话，看有没有什么有用信息(JSON):
        {conversations_json}

        当前长期/中期记忆(JSON)，可用于直接回答用户关于个人资料、近期饮食和训练反馈的问题：
        {memory_context_json}

        可用工具及参数 schema(JSON):
        {tool_descriptions_json}

        判断是否需要调用工具来完成任务。
        规则：
        - 如果用户在询问“我叫什么”“我的身高是多少”“我的体重是多少”“我最近吃了什么”这类可直接从记忆回答的问题，优先直接用记忆回答，不调用工具。
        - 如果用户要求根据身体状态、训练强度、可用时间、饮食限制制定饮食计划，优先调用 diet_plan_generator。
        - 如果用户询问某个城市今天/明天/未来几天的天气，或询问某天气是否适合运动、跑步、健身，优先调用 weather_fitness_advisor。
        - 对天气相关问题，优先把自然语言城市名放入 weather_fitness_advisor 的 city 参数；when 根据“今天/明天”选择 today 或 tomorrow。
        - 如果用户要求规划附近适合的跑步路线、晨跑路线、夜跑路线、5公里/10公里跑步路线，优先调用 running_route_advisor。
        - 对跑步路线相关问题，优先把自然语言地点放入 running_route_advisor 的 start_location；若用户提到城市，也可填 city；若提到 3公里/5公里/10公里等距离，填 target_distance_km。
        - 如果用户询问有哪些训练部位、身体部位列表、可训练的身体区域，优先调用 rapidapi_bodyparts。
        - 如果需要工具：
          - tool_name 必须严格等于可用工具名之一，不允许添加任何前缀或后缀。
          - args 必须严格满足该工具的参数 schema。
          - tavily_search 的 args 必须包含 query。
          - diet_plan_generator 的 args 应优先传 user_profile，并尽量从当前记忆中补齐 height_cm、weight_kg、body_condition、goal、exercise_intensity、available_cooking_time_minutes、diet、intolerances、preferred_ingredients、disliked_ingredients、preferred_cuisines、daily_diet。
          - result 必须为 null。
        - 如果不需要工具：
          - tool_calls 必须为空数组。
          - result 直接给出该子任务结果。
        
        严格输出一个 JSON 对象，不要 Markdown：
        - 只能使用 JSON 的 null / true / false，不要使用 Python 的 None / True / False。
        - 字符串必须使用双引号。
        - 不要在 JSON 外输出解释文字。
        {{          
            "tool_calls": [
                {{
                    "tool_name": "工具名",
                    "args": {{}},
                    "id": "..."
                }},
                {{
                    "tool_name": "...",
                    "args": {{}},
                    "id": "..."
                }},
            ],
            "result": null
        }}
        """

        prompt_observation = f"""
        用户原始问题: {user_message}
        第一轮 AI 分析: {first_ai_message}
        当前识别意图: {state.reasoning.intent}
        你正在执行任务: {task.name}
        任务描述: {task.description}    
        
        工具调用记录如下:
        {[call.model_dump() for call in task.tool_calls]}
        工具结果如下:
        {task.tool_results}

        如果工具全部失败，请基于已有信息给出保守结果，并说明缺少哪些信息。
        
        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "result": "子任务结果"
        }}
        """

        if not task.tool_calls:
            try:
                data = self.invoke_json(prompt_reason)
            except LLMJsonParseError as exc:
                data = self._fallback_reason_data(state, task, str(exc))
            task.tool_calls = self._parse_tool_calls(
                data.get("tool_calls"), available_tools, user_message, task
            )
            task.result = data.get("result")

            if task.tool_calls and not task.result:
                task.status = TaskStatus.waiting_for_tool

                print("=" * 20)
                print("ReasonNode - Tool Calls")
                print("=" * 20)
                print(data)

                return state
        else:
            try:
                data = self.invoke_json(prompt_observation)
            except LLMJsonParseError as exc:
                data = {
                    "result": f"工具结果已获取，但模型解析工具观察结果时失败：{exc}。请根据已有工具结果保守生成回答。"
                }
            task.result = data.get("result")

        if task.result:
            task.status = TaskStatus.done
            state.reasoning.advance_task()

            print("=" * 20)
            print("ReasonNode - Task Result")
            print("=" * 20)
            print(data)

            return state

        task.status = TaskStatus.failed
        task.error = "Reasoning step did not produce a task result."
        state.reasoning.errors.append(f"{task.name}: {task.error}")

        print("=" * 20)
        print("ReasonNode - Failed Task")
        print("=" * 20)
        print(data)

        return state

    @staticmethod
    def _memory_context(state) -> dict:
        long_term = state.memory.long_term_memory
        return {
            "name": long_term.name,
            "gender": long_term.gender,
            "job": long_term.job,
            "physical_profile": long_term.physical_profile.model_dump(),
            "lifestyle_profile": long_term.lifestyle_profile.model_dump(),
            "dietary_profile": long_term.dietary_profile.model_dump(),
            "daily_diet": state.memory.mid_term_memory.daily_diet,
            "training_feedbacks": state.memory.mid_term_memory.training_feedbacks,
        }

    @staticmethod
    def _fallback_reason_data(state, task: Task, error_message: str) -> dict:
        user_message = str(ReasonNode.latest_user_text(state))
        available_tools = set(state.tools.available_tools.keys())
        is_diet_plan_request = any(
            keyword in user_message
            for keyword in ["饮食", "吃", "食谱", "增肌期间", "减脂期间", "控制饮食"]
        )
        is_running_route_request = any(
            keyword in user_message
            for keyword in ["跑步路线", "跑步", "晨跑", "夜跑", "路线规划", "公里路线", "适合跑步"]
        )
        is_bodyparts_request = any(
            keyword in user_message
            for keyword in ["训练部位", "身体部位", "锻炼部位", "部位列表", "有哪些部位", "可练部位"]
        )

        if is_bodyparts_request and "rapidapi_bodyparts" in available_tools:
            return {
                "tool_calls": [
                    {
                        "tool_name": "rapidapi_bodyparts",
                        "args": ReasonNode._repair_tool_args(
                            "rapidapi_bodyparts",
                            {},
                            user_message,
                            task,
                        ),
                        "id": "fallback-bodyparts",
                    }
                ],
                "result": None,
                "fallback_reason": error_message,
            }

        if is_running_route_request and "running_route_advisor" in available_tools:
            return {
                "tool_calls": [
                    {
                        "tool_name": "running_route_advisor",
                        "args": ReasonNode._repair_tool_args(
                            "running_route_advisor",
                            {},
                            user_message,
                            task,
                        ),
                        "id": "fallback-running-route",
                    }
                ],
                "result": None,
                "fallback_reason": error_message,
            }

        if is_diet_plan_request and "diet_plan_generator" in available_tools:
            long_term = state.memory.long_term_memory
            user_profile = {
                "name": long_term.name,
                "gender": long_term.gender,
                "job": long_term.job,
                "height_cm": long_term.physical_profile.height_cm,
                "weight_kg": long_term.physical_profile.weight_kg,
                "age": long_term.physical_profile.age,
                "body_condition": long_term.physical_profile.body_condition,
                "goal": state.reasoning.extracted_info.get("profile", {}).get("goal")
                or long_term.lifestyle_profile.goal
                or "增肌",
                "activity_level": long_term.lifestyle_profile.activity_level,
                "exercise_intensity": long_term.lifestyle_profile.exercise_intensity,
                "available_cooking_time_minutes": long_term.lifestyle_profile.available_cooking_time_minutes,
                "diet": long_term.dietary_profile.diet,
                "intolerances": long_term.dietary_profile.intolerances,
                "preferred_cuisines": long_term.dietary_profile.preferred_cuisines,
                "preferred_ingredients": long_term.dietary_profile.preferred_ingredients,
                "disliked_ingredients": long_term.dietary_profile.disliked_ingredients,
                "daily_diet": state.memory.mid_term_memory.daily_diet,
            }
            return {
                "tool_calls": [
                    {
                        "tool_name": "diet_plan_generator",
                        "args": {
                            "user_profile": user_profile,
                            "meal_count": 3,
                            "number_per_meal": 2,
                            "goal": user_profile["goal"],
                        },
                        "id": "fallback-diet-plan",
                    }
                ],
                "result": None,
                "fallback_reason": error_message,
            }

        return {
            "tool_calls": [],
            "result": "我已读取你的历史信息，但当前模型输出格式异常。请你稍后再试，或补充目标、时间和饮食偏好后我再给出更具体建议。",
            "fallback_reason": error_message,
        }

    @staticmethod
    def _parse_tool_calls(
        raw_calls,
        available_tools: list[str],
        user_message: str,
        task: Task,
    ) -> list[ToolCall]:
        if not isinstance(raw_calls, list):
            return []

        parsed: list[ToolCall] = []
        available = set(available_tools)
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("tool_name") or raw_call.get("name") or "").strip()
            if name.startswith("functions."):
                name = name.removeprefix("functions.")
            if not name or name not in available:
                continue
            args = raw_call.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            args = ReasonNode._repair_tool_args(name, args, user_message, task)
            parsed.append(
                ToolCall(
                    id=raw_call.get("id"),
                    name=name,
                    args=args,
                )
            )

        return parsed

    @staticmethod
    def _repair_tool_args(
        tool_name: str,
        args: dict,
        user_message: str,
        task: Task,
    ) -> dict:
        if tool_name == "tavily_search" and not args.get("query"):
            query_parts = [user_message, task.name, task.description]
            args["query"] = " ".join(str(part) for part in query_parts if part)

        if tool_name == "rapidapi_bodyparts" and not args.get("query_params"):
            args["query_params"] = {}

        if tool_name == "weather_fitness_advisor":
            if not args.get("city"):
                city_match = re.search(r"([\u4e00-\u9fff]{2,12}?)(?:今天|明天|天气)", user_message)
                if city_match:
                    args["city"] = city_match.group(1)
            if not args.get("when"):
                args["when"] = "tomorrow" if "明天" in user_message else "today"

        if tool_name == "running_route_advisor":
            normalized_message = user_message.replace("，", " ").replace("。", " ").strip()

            if not args.get("start_location"):
                location_match = re.search(r"(?:我在|在)(.+?)(?:附近|周边|帮我|请|想|需要|，|。|$)", normalized_message)
                if location_match:
                    args["start_location"] = location_match.group(1).strip()
                else:
                    cleaned = re.sub(r"帮我|请|规划|推荐|设计|安排|一条|一个|合适的|适合的|附近的", "", normalized_message)
                    cleaned = re.sub(r"(晨跑|夜跑|跑步)路线", "", cleaned)
                    cleaned = re.sub(r"(晨跑|夜跑|跑步)", "", cleaned)
                    cleaned = re.sub(r"\d+(?:\.\d+)?\s*公里", "", cleaned)
                    cleaned = cleaned.strip(" ，。,.？?在")
                    args["start_location"] = cleaned or user_message

            if not args.get("city"):
                city_match = re.search(r"([\u4e00-\u9fff]{2,12}?(?:市|区|县))", normalized_message)
                if city_match:
                    args["city"] = city_match.group(1)
                else:
                    start_location = str(args.get("start_location") or "")
                    fallback_city_match = re.search(r"([\u4e00-\u9fff]{2,12}?(?:市|区|县))", start_location)
                    if fallback_city_match:
                        args["city"] = fallback_city_match.group(1)

            if not args.get("target_distance_km"):
                distance_match = re.search(r"(\d+(?:\.\d+)?)\s*公里", user_message)
                if distance_match:
                    args["target_distance_km"] = float(distance_match.group(1))
                else:
                    args["target_distance_km"] = 5.0

            if not args.get("route_preference"):
                if "绿道" in user_message or "江边" in user_message or "湖边" in user_message:
                    args["route_preference"] = "greenway"
                elif "操场" in user_message or "田径场" in user_message:
                    args["route_preference"] = "track"
                elif "公园" in user_message or "环线" in user_message or "湖" in user_message:
                    args["route_preference"] = "park_loop"
                else:
                    args["route_preference"] = "general"

        return args

    @staticmethod
    def _answer_from_memory(state) -> str | None:
        user_message = str(ReasonNode.latest_user_text(state)).strip().lower()
        long_term = state.memory.long_term_memory
        physical = long_term.physical_profile
        lifestyle = long_term.lifestyle_profile
        dietary = long_term.dietary_profile
        mid_term = state.memory.mid_term_memory

        if any(keyword in user_message for keyword in ["我叫什么", "我的名字", "我叫啥"]):
            if long_term.name:
                return f"你叫{long_term.name}。"
            return None

        if "身高" in user_message and any(
            keyword in user_message for keyword in ["是多少", "多高", "记得", "记录"]
        ):
            if physical.height_cm is not None:
                return f"你之前记录的身高是{physical.height_cm}厘米。"
            return None

        if any(keyword in user_message for keyword in ["体重是多少", "我多重", "多少公斤", "记得我体重", "记录的体重"]):
            if physical.weight_kg is not None:
                return f"你之前记录的体重是{physical.weight_kg}公斤。"
            return None

        if any(keyword in user_message for keyword in ["我最近吃了什么", "我今天吃了什么", "我的饮食记录"]):
            if mid_term.daily_diet:
                return f"你最近记录的饮食包括：{'、'.join(mid_term.daily_diet)}。"
            return None

        if any(keyword in user_message for keyword in ["我的目标", "健身目标"]):
            if lifestyle.goal:
                return f"你当前记录的目标是{lifestyle.goal}。"
            return None

        if any(keyword in user_message for keyword in ["我不能吃什么", "我有什么忌口", "我有什么不耐受"]):
            if dietary.intolerances or dietary.disliked_ingredients:
                parts = []
                if dietary.intolerances:
                    parts.append(f"不耐受：{'、'.join(dietary.intolerances)}")
                if dietary.disliked_ingredients:
                    parts.append(f"不喜欢：{'、'.join(dietary.disliked_ingredients)}")
                return f"你当前记录的饮食限制有：{'；'.join(parts)}。"
            return None

        return None
