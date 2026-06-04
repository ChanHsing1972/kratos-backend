"""任务推理与工具决策节点。

ReasonNode 面向当前子任务判断是否需要工具：可直接回答则写入任务结果；
需要工具则产出 `ToolCall` 并进入等待工具状态；已有工具结果时负责汇总观察结果。
工具参数修复和模型输出 fallback 放在 `app.agent.tool_planner`，避免节点本身膨胀。
"""

import json
import re

from app.agent.intent_policy import (
    INFO_INTENTS,
    has_plan_intent,
    information_tool_calls,
    summarize_tool_result,
)
from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import TaskStatus
from app.agent.tool_planner import build_fallback_reason_data, parse_tool_calls, repair_tool_args


class ReasonNode(BaseNode):
    """执行当前子任务的一步推理。"""

    def __call__(self, state):
        """推进当前任务状态，可能生成工具调用、任务结果或失败原因。"""

        task = state.reasoning.current_task()
        if task is None:
            return state

        memory_answer = self._answer_from_memory(state)
        if memory_answer is not None:
            task.status = TaskStatus.done
            task.result = memory_answer
            state.reasoning.advance_task()

            self.logger.debug("ReasonNode memory result: %s", {"result": memory_answer})

            return state

        available_tools = sorted(state.tools.available_tools.keys())
        user_message = self.latest_user_text(state)
        task.status = TaskStatus.running
        fast_data = self._deterministic_reason_data(state, task, user_message, available_tools)
        if fast_data is not None:
            return self._apply_reason_data(state, task, fast_data, available_tools, user_message)

        convs = state.conversation.conversations
        tool_descriptions = self.describe_tools(state.tools.available_tools)
        first_ai_message = state.conversation.first_ai_message or state.result.first_response or ""
        extracted_info = state.reasoning.extracted_info or {}
        memory_context = self._memory_context(state)
        skill_context = self.describe_active_skills(state)
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

        启用 Skill:
        {skill_context}
        
        查询工具之前，先查看过往会话，看有没有什么有用信息(JSON):
        {conversations_json}

        当前长期/中期记忆(JSON)，可用于直接回答用户关于个人资料、近期饮食和训练反馈的问题：
        {memory_context_json}

        记忆优先级规则：
        - 工作记忆优先级最高，只要仍与当前任务相关，应优先约束本轮回答。
        - 短期记忆次之，适合解释用户最近位置、近期需求、近期阶段安排。
        - 长期记忆再次之，适合提供稳定目标、偏好、长期限制与长期风险背景。
        - 若三层记忆冲突：工作记忆 > 短期记忆 > 长期记忆。
        - 若数据库结构化上下文与记忆点冲突，以数据库已确认上下文优先；但可在回答中说明用户近期表达的临时变化。

        可用工具及参数 schema(JSON):
        {tool_descriptions_json}

        判断是否需要调用工具来完成任务。
        规则：
        - Skill 是领域能力包，不是代码执行插件；你只能遵守其策略、工具范围、输出格式和禁忌规则。
        - 如果启用 Skill 声明了可用工具，当前工具列表已经按这些 Skill 做了范围约束。
        - 严禁编造用户资料；年龄、身高、体重、目标、训练经验等只能来自“已提取关键信息”或“当前长期/中期记忆”。
        - 字段为 null、None、空字符串或未出现时，必须视为未知，不得自行填充。
        - 计算 BMR、热量或 1RM 时，如果缺少必需的性别、年龄、身高、体重、重量或次数，不要用默认值硬算；直接说明缺少哪些信息。
        - 严格区分“60分钟”和“60岁”：available_time_minutes 或“60分钟”只表示训练时长，绝不能当作年龄。
        - 如果任务描述与已提取信息冲突，以已提取信息和数据库记忆为准，并在结果中纠正，不要沿用错误任务描述。
        - 如果用户在询问“我叫什么”“我的身高是多少”“我的体重是多少”“我最近吃了什么”这类可直接从记忆回答的问题，优先直接用记忆回答，不调用工具。
        - 如果用户要求根据身体状态、训练强度、可用时间、饮食限制制定饮食计划，优先调用 diet_plan_generator。
        - 如果当前长期/中期记忆中的 database_context.knowledge_base 存在相关外部知识，子任务结果需要保留对应 citation，例如 [知识库:膝痛训练安全#1]。
        - 如果用户提到急性疼痛、膝盖/腰/肩不适、极度疲劳，必须优先调用 pain_safety_gate，再决定是否替代训练或休息。
        - 如果用户要求替换某个动作，或安全分流结果显示需要 modify_plan/rest_or_recovery，优先调用 exercise_substitution_advisor 给出更安全替代动作。
        - 如果用户要求按训练时长安排组数/动作数量，优先调用 calculate_workout_volume。
        - 如果用户要求计算基础代谢或每日消耗，优先调用 calculate_bmr。
        - 如果用户要求估算 1RM 或训练重量，优先调用 estimate_1rm。
        - 如果用户要求估算运动消耗或为了消耗目标热量需要运动多久，优先调用 calculate_calories_burned。
        - 如果用户询问某个城市今天/明天/未来几天的天气，或询问某天气是否适合运动、跑步、健身，优先调用 weather_fitness_advisor。
        - 对天气相关问题，优先把自然语言城市名放入 weather_fitness_advisor 的 city 参数；when 根据“今天/明天”选择 today 或 tomorrow。
        - 如果用户要求规划附近适合的跑步路线、晨跑路线、夜跑路线、5公里/10公里跑步路线，优先调用 running_route_advisor。
        - 对跑步路线相关问题，优先把自然语言地点放入 running_route_advisor 的 start_location；若用户提到城市，也可填 city；若提到 3公里/5公里/10公里等距离，填 target_distance_km。
        - 如果用户询问有哪些训练部位、身体部位列表、可训练的身体区域，优先调用 rapidapi_bodyparts。
        - 如果需要工具：
          - tool_name 必须严格等于可用工具名之一，不允许添加任何前缀或后缀。
          - args 必须严格满足该工具的参数 schema。
          - 工具会在执行前进行参数校验；缺少必填字段或数值越界时，本轮会得到 validation_failed 降级结果，所以你应尽量从用户问题和记忆中补齐必要参数。
          - 如果工具返回 fallback=true，必须把它视为保守降级结果，不要当作精确外部 API 结果；最终回答要说明不确定性。
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

        启用 Skill:
        {skill_context}
        
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
                data = self.invoke_json(prompt_reason, state)
            except Exception as exc:  # noqa: BLE001
                data = build_fallback_reason_data(state, task, user_message, str(exc))
            task.tool_calls = parse_tool_calls(
                data.get("tool_calls"), available_tools, user_message, task, state
            )
            task.result = data.get("result")

            if task.tool_calls and not task.result:
                task.status = TaskStatus.waiting_for_tool

                self.logger.debug("ReasonNode tool calls: %s", data)

                return state
        else:
            try:
                data = self.invoke_json(prompt_observation, state)
            except Exception as exc:  # noqa: BLE001
                data = {
                    "result": f"工具结果已获取，但模型解析工具观察结果时失败：{exc}。请根据已有工具结果保守生成回答。"
                }
            task.result = data.get("result")

        if task.result:
            task.status = TaskStatus.done
            state.reasoning.advance_task()

            self.logger.debug("ReasonNode task result: %s", data)

            return state

        task.status = TaskStatus.failed
        task.error = "Reasoning step did not produce a task result."
        state.reasoning.errors.append(f"{task.name}: {task.error}")

        self.logger.debug("ReasonNode failed task: %s", data)

        return state

    def _apply_reason_data(
        self,
        state,
        task,
        data: dict,
        available_tools: list[str],
        user_message: str,
    ):
        if not task.tool_calls:
            task.tool_calls = parse_tool_calls(
                data.get("tool_calls"), available_tools, user_message, task, state
            )
            task.result = data.get("result")
            if task.tool_calls and not task.result:
                task.status = TaskStatus.waiting_for_tool
                self.logger.debug("ReasonNode deterministic tool calls: %s", data)
                return state
        else:
            task.result = data.get("result")

        if task.result:
            task.status = TaskStatus.done
            state.reasoning.advance_task()
            self.logger.debug("ReasonNode deterministic result: %s", data)
            return state

        task.status = TaskStatus.failed
        task.error = "Reasoning step did not produce a task result."
        state.reasoning.errors.append(f"{task.name}: {task.error}")
        return state

    @staticmethod
    def _deterministic_reason_data(
        state,
        task,
        user_message: str,
        available_tools: list[str],
    ) -> dict | None:
        available = set(available_tools)
        is_fitness_plan = any(intent in state.reasoning.intent for intent in ["健身计划", "调整计划"])
        is_information_query = any(intent in INFO_INTENTS for intent in state.reasoning.intent)
        is_diet_record = "饮食记录" in state.reasoning.intent
        is_diet_plan = "饮食计划" in state.reasoning.intent and not is_diet_record

        if task.tool_calls:
            if is_information_query and not is_fitness_plan:
                summaries = [summarize_tool_result(result) for result in task.tool_results]
                return {
                    "tool_calls": [],
                    "result": "；".join(item for item in summaries if item) or "工具结果已获取，请据此回答用户的信息查询。",
                }
            if is_fitness_plan:
                summaries = []
                for result in task.tool_results:
                    if not isinstance(result, dict):
                        summaries.append(str(result))
                        continue
                    tool = result.get("tool")
                    if tool == "pain_safety_gate":
                        reasons = "；".join(str(item) for item in result.get("reasons") or [])
                        summaries.append(f"安全门建议：{result.get('action')}，{reasons}")
                    elif tool == "calculate_workout_volume":
                        summaries.append(
                            "训练量估算："
                            f"{result.get('exercise_count')} 个动作，"
                            f"每个动作约 {result.get('sets_per_exercise')} 组，"
                            f"预计用时 {result.get('estimated_used_minutes')} 分钟。"
                        )
                    else:
                        summaries.append(json.dumps(result, ensure_ascii=False, default=str))
                return {
                    "tool_calls": [],
                    "result": "；".join(item for item in summaries if item) or "工具结果已获取，请据此生成保守训练建议。",
                }
            return None

        tool_calls: list[dict] = []
        if is_diet_record:
            return {
                "tool_calls": [],
                "result": ReasonNode._diet_record_result(state),
            }

        if is_diet_plan:
            if "diet_plan_generator" in available:
                return build_fallback_reason_data(
                    state,
                    task,
                    user_message,
                    "structured_diet_plan_fast_path",
                )
            return {
                "tool_calls": [],
                "result": "已读取你的饮食目标和身体上下文，但当前没有可用的饮食计划工具；请基于已有信息给出保守饮食建议。",
            }

        if is_information_query and not is_fitness_plan:
            tool_calls = information_tool_calls(state, task, user_message, available_tools)
            if tool_calls:
                return {"tool_calls": tool_calls, "result": None}
            return {
                "tool_calls": [],
                "result": "未找到可用的查询工具，请基于已有上下文回答，并说明实时信息可能无法获取。",
            }

        if is_fitness_plan:
            if "pain_safety_gate" in available:
                tool_calls.append(
                    {
                        "tool_name": "pain_safety_gate",
                        "args": repair_tool_args("pain_safety_gate", {}, user_message, task, state),
                        "id": "deterministic-safety-gate",
                    }
                )
            if "calculate_workout_volume" in available:
                tool_calls.append(
                    {
                        "tool_name": "calculate_workout_volume",
                        "args": repair_tool_args("calculate_workout_volume", {}, user_message, task, state),
                        "id": "deterministic-workout-volume",
                    }
                )
            if tool_calls:
                return {"tool_calls": tool_calls, "result": None}
            return {
                "tool_calls": [],
                "result": "已读取用户上下文和训练目标，请生成可执行训练安排，并说明必要的安全边界。",
            }

        return None

    @staticmethod
    def _diet_record_result(state) -> str:
        extracted = state.reasoning.extracted_info or {}
        daily_diet = extracted.get("daily_diet") if isinstance(extracted, dict) else []
        daily_items = [str(item).strip() for item in daily_diet or [] if str(item).strip()]
        has_attachment = ReasonNode._has_latest_attachment(state)

        if daily_items:
            return (
                "用户提供了待记录餐食："
                + "；".join(daily_items)
                + "。请整理为可确认的饮食记录；如缺少热量或三大营养素，明确说明需要用户补充分量或图片。"
            )
        if has_attachment:
            return "用户上传了餐食附件。请基于可见食物估算热量和三大营养素，并生成需要用户确认后保存的饮食记录。"
        return "用户想记录饮食，但尚未提供具体食物、份量或图片。请先请用户补充餐食明细，暂不生成可保存记录。"

    @staticmethod
    def _has_latest_attachment(state) -> bool:
        if getattr(state.result, "user_attachments", None):
            return True
        for message in reversed(state.conversation.messages):
            if getattr(message, "type", None) != "human":
                continue
            content = getattr(message, "content", None)
            return isinstance(content, list) and any(
                isinstance(item, dict) and item.get("type") in {"image_url", "file"}
                for item in content
            )
        return False

    @staticmethod
    def _memory_context(state) -> dict:
        """整理 ReasonNode prompt 所需的长期/中期记忆和数据库上下文。"""

        long_term = state.memory.long_term_memory
        return {
            "name": long_term.name,
            "gender": long_term.gender,
            "job": long_term.job,
            "location": long_term.location,
            "physical_profile": long_term.physical_profile.model_dump(),
            "lifestyle_profile": long_term.lifestyle_profile.model_dump(),
            "dietary_profile": long_term.dietary_profile.model_dump(),
            "long_term_memory_points": [
                item.model_dump(mode="json")
                for item in state.memory.long_term_memory_points
            ],
            "short_term_memory_points": [
                item.model_dump(mode="json")
                for item in state.memory.short_term_memory_points
            ],
            "working_memory_points": [
                item.model_dump(mode="json")
                for item in state.memory.working_memory_points
            ],
            "daily_diet": state.memory.mid_term_memory.daily_diet,
            "training_feedbacks": state.memory.mid_term_memory.training_feedbacks,
            "database_context": state.memory.database_context,
            "turn_summaries": [item.model_dump() for item in state.memory.turn_summaries],
            "conversation_summaries": state.conversation.summaries,
        }

    @staticmethod
    def _answer_from_memory(state) -> str | None:
        """对纯记忆查询进行短路回答，避免无意义工具调用。"""

        user_message = str(ReasonNode.latest_user_text(state)).strip().lower()
        if has_plan_intent(state.reasoning.intent):
            return None

        if re.search(r"(工具|tools?|可调用|能力清单|支持哪些)", user_message, flags=re.IGNORECASE):
            tool_descriptions = ReasonNode.describe_tools(state.tools.available_tools)
            if not tool_descriptions:
                return "当前没有启用可调用工具。"
            lines = ["当前可调用工具如下："]
            for tool in tool_descriptions:
                description = str(tool.get("description") or "暂无描述").strip()
                lines.append(f"- `{tool.get('name')}`：{description}")
            return "\n".join(lines)

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

        if any(keyword in user_message for keyword in ["体脂", "体脂率"]):
            if physical.body_fat_percentage is not None:
                return f"你最近记录的体脂率是{physical.body_fat_percentage}%。"
            return None

        if any(keyword in user_message for keyword in ["睡眠", "睡多久", "睡了多久"]):
            if physical.sleep_hours is not None:
                return f"你最近记录的睡眠时长是{physical.sleep_hours}小时。"
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
