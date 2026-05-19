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

        data = self.invoke_json(prompt)
        tasks_data = data.get("tasks") or []
        tasks: list[Task] = []
        for index, raw_task in enumerate(tasks_data):
            if not isinstance(raw_task, dict):
                continue
            raw_task["task_id"] = int(raw_task.get("task_id", index))
            raw_task["name"] = str(raw_task.get("name") or f"任务 {index + 1}")
            raw_task["description"] = str(raw_task.get("description") or raw_task["name"])
            tasks.append(Task(**raw_task))

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
        print(data)

        return state
