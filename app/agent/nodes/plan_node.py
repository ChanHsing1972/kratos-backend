from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task


class PlanNode(BaseNode):
    def __call__(self, state):
        intent = state.reasoning.intent
        user_msg = self.latest_user_text(state)
        reflection = state.reasoning.reflection or {}

        prompt = f"""
        你是健身 Agent 的任务规划器。根据用户意图和消息拆解任务。
        要求：
        - 任务数量控制在 1 到 5 个。
        - 每个任务必须能独立执行。
        - 如果有反思建议，请在新计划中修正问题。
        - task_id 从 0 开始递增。
        
        意图: {intent}
        用户消息: {user_msg}
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
            raw_task["description"] = raw_task.get("description") or raw_task["name"]
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
