"""任务规划节点。

PlanNode 把意图、用户消息、数据库上下文和反思建议拆成 1-5 个可执行子任务。
它不调用工具，只产出任务队列；工具选择由 ReasonNode 负责。
"""

from app.agent.nodes.base_node import BaseNode
from app.agent.intent_policy import (
    INFO_INTENTS,
    has_plan_intent,
    is_news_query,
    is_route_query,
    is_weather_query,
)
from app.agent.state.reasoning import Task


class PlanNode(BaseNode):
    """根据当前意图生成或重生成任务队列。"""

    def __call__(self, state):
        """更新 `state.reasoning.tasks` 并把当前任务指针归零。"""

        intent = state.reasoning.intent
        user_msg = self.latest_user_text(state)
        first_ai_message = state.conversation.first_ai_message or state.result.first_response or ""
        reflection = state.reasoning.reflection or {}
        extracted_info = state.reasoning.extracted_info or {}
        skill_context = self.describe_active_skills(state)
        deterministic_tasks = self._deterministic_tasks(intent, user_msg, extracted_info)
        if deterministic_tasks is not None and not reflection:
            state.reasoning.tasks = deterministic_tasks
            state.reasoning.current_task_index = 0
            state.reasoning.need_replan = False
            self.logger.debug("PlanNode deterministic result: %s", {"tasks": [task.model_dump() for task in deterministic_tasks]})
            return state

        prompt = f"""
        你是健身 Agent 的任务规划器。请根据用户意图、用户消息、第一轮 AI 分析以及反思建议来拆解任务。
        要求：
        - 任务数量控制在 1 到 5 个。
        - 每个任务必须能独立执行。
        - 如果有反思建议，请在新计划中修正问题。
        - 如果启用了 Skill，任务拆解必须遵守 Skill 的适用场景、提示片段和禁忌规则。
        - task_id 从 0 开始递增。
        - 任务名称简洁明确，描述写清楚要完成什么。
        - 严禁编造用户资料；已提取关键信息里为 None/null/空值的年龄、身高、体重、经验等，不得写入任务描述。
        - 严格区分“60分钟”和“60岁”：available_time_minutes 或用户说“60分钟”只能表示训练时长，不能推断为年龄。
        - 如果年龄未知，用“年龄未提供”；如果用户只说“训练两个月”，只能表示训练经验，不能推断年龄。

        启用 Skill:
        {skill_context}

        用户意图: {intent}
        用户消息: {user_msg}
        第一轮 AI 分析: {first_ai_message}
        已提取关键信息: {extracted_info}
        反思建议: {reflection}

        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "tasks": [
                {{
                    "task_id": 0,
                    "name": "任务名称",
                    "description": "任务描述"
                }}
            ]
        }}
        """

        data = self.invoke_json(prompt, state)
        tasks_data = data.get("tasks") or []
        tasks: list[Task] = []
        for index, raw_task in enumerate(tasks_data):
            if not isinstance(raw_task, dict):
                continue
            raw_task["task_id"] = int(raw_task.get("task_id", index))
            raw_task["name"] = str(raw_task.get("name") or f"任务 {index + 1}")
            raw_task["description"] = str(raw_task.get("description") or raw_task["name"])
            raw_task["description"] = self._sanitize_task_description(
                raw_task["description"],
                extracted_info,
            )
            tasks.append(Task(**raw_task))

        tasks = self._ensure_actionable_fitness_plan_task(tasks, intent, user_msg)

        if not tasks:
            tasks = [
                Task(
                    task_id=0,
                    name="直接回答用户问题",
                    description=user_msg,
                )
            ]

        state.reasoning.tasks = tasks
        state.reasoning.current_task_index = 0
        state.reasoning.need_replan = False

        self.logger.debug("PlanNode result: %s", {"tasks": [task.model_dump() for task in tasks]})

        return state

    @staticmethod
    def _deterministic_tasks(
        intent: list[str],
        user_message: str,
        extracted_info: dict,
    ) -> list[Task] | None:
        if any(item in INFO_INTENTS for item in intent) and not has_plan_intent(intent):
            tasks: list[Task] = []
            if "天气查询" in intent or is_weather_query(user_message):
                tasks.append(
                    Task(
                        task_id=len(tasks),
                        name="查询天气",
                        description="查询用户提到地点和时间对应的天气，并整理对运动安排的简要影响。",
                    )
                )
            if "新闻搜索" in intent or is_news_query(user_message):
                tasks.append(
                    Task(
                        task_id=len(tasks),
                        name="搜索相关新闻",
                        description="搜索用户请求主题的近期资讯，保留标题、摘要和可靠链接。",
                    )
                )
            if "路线查询" in intent or is_route_query(user_message):
                tasks.append(
                    Task(
                        task_id=len(tasks),
                        name="查询路线",
                        description="根据用户提到的位置、距离和偏好查询适合的路线建议。",
                    )
                )
            if not tasks:
                tasks.append(
                    Task(
                        task_id=0,
                        name="直接回答信息查询",
                        description="基于用户问题、会话历史和数据库上下文回答，不生成训练计划。",
                    )
                )
            return tasks

        if "健身计划" in intent:
            profile = extracted_info.get("profile", {}) if isinstance(extracted_info, dict) else {}
            available_time = profile.get("available_time_minutes") if isinstance(profile, dict) else None
            is_cycle = any(keyword in user_message for keyword in ["每周", "一周", "下周", "周期", "长期", "多周", "周计划"])
            title = "生成周期训练计划" if is_cycle else "生成今日训练计划"
            time_part = f"可用训练时长约 {available_time} 分钟。" if available_time else "结合档案中的可训练时长。"
            return [
                Task(
                    task_id=0,
                    name=title,
                    description=(
                        "基于已读取的数据库上下文、当前恢复状态和启用 Skill 的安全约束，"
                        f"{'生成至少一周的训练安排。' if is_cycle else '生成本次可直接执行的训练安排。'}"
                        f"{time_part}"
                    ),
                )
            ]

        if "饮食计划" in intent:
            return [
                Task(
                    task_id=0,
                    name="生成饮食建议",
                    description="基于用户目标、身体数据、饮食记录和当前输入生成可执行饮食建议。",
                )
            ]

        if "反馈" in intent or "调整计划" in intent:
            return [
                Task(
                    task_id=0,
                    name="处理训练反馈",
                    description="读取用户反馈和近期训练上下文，给出安全调整建议。",
                )
            ]

        return None

    @staticmethod
    def _sanitize_task_description(description: str, extracted_info: dict) -> str:
        """移除模型把训练时长误写成年龄的高风险描述。"""

        profile = extracted_info.get("profile") if isinstance(extracted_info, dict) else {}
        if isinstance(profile, dict) and profile.get("age") is not None:
            return description

        return (
            description.replace("60岁、", "")
            .replace("60岁", "年龄未提供")
            .replace("高龄", "")
            .replace("老年", "")
        )

    @staticmethod
    def _ensure_actionable_fitness_plan_task(
        tasks: list[Task],
        intent: list[str],
        user_message: str,
    ) -> list[Task]:
        """计划类请求必须至少包含一个真正生成训练内容的任务。"""

        if "健身计划" not in intent:
            return tasks

        plan_keywords = ["制定", "生成", "安排", "训练内容", "主训练"]
        has_plan_task = any(
            any(keyword in f"{task.name} {task.description}" for keyword in plan_keywords)
            and "收集" not in task.name
            for task in tasks
        )
        if has_plan_task:
            return tasks

        actionable_task = Task(
            task_id=len(tasks),
            name="生成今日训练计划",
            description=(
                "基于已读取的数据库上下文和用户当前请求，生成今日可执行训练计划。"
                "如果档案信息不完整，先采用保守默认强度并在结果中说明缺口；"
                "不要只要求用户补充信息。"
            ),
        )
        return [*tasks, actionable_task] if tasks else [actionable_task]
