from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task


class PlanNode(BaseNode):
    def __call__(self, state):
        intent = state.reasoning.intent
        user_msg = self.latest_user_text(state)
        first_ai_message = state.conversation.first_ai_message or state.result.first_response or ""
        reflection = state.reasoning.reflection or {}
        extracted_info = state.reasoning.extracted_info or {}
        skill_context = self.describe_active_skills(state)

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

        # 禁止删除以下打印语句
        print("=" * 20)
        print("PlanNode")
        print("=" * 20)
        print({"tasks": [task.model_dump() for task in tasks]})

        return state

    @staticmethod
    def _sanitize_task_description(description: str, extracted_info: dict) -> str:
        if extracted_info.get("age") is not None:
            return description

        return (
            description
            .replace("60岁、", "")
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
        if "健身计划" not in intent:
            return tasks

        has_plan_task = any(
            any(keyword in f"{task.name} {task.description}" for keyword in ["制定", "生成", "安排", "训练内容", "主训练"])
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
