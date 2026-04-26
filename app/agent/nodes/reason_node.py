from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task, TaskStatus
from app.agent.state.tools import ToolCall


class ReasonNode(BaseNode):

    def __call__(self, state):
        task = state.reasoning.current_task()
        if task is None:
            return state

        convs = state.conversation.conversations
        available_tools = sorted(state.tools.available_tools.keys())
        tool_descriptions = self.describe_tools(state.tools.available_tools)
        user_message = self.latest_user_text(state)

        task.status = TaskStatus.running

        prompt_reason = f"""        
        用户原始问题: {user_message}
        你正在执行任务: {task.name}
        任务描述: {task.description}
        
        查询工具之前，先查看过往会话，看有没有什么有用信息:
        {convs}

        可用工具及参数 schema:
        {tool_descriptions}

        判断是否需要调用工具来完成任务。
        如果需要工具：
        - tool_name 必须严格等于可用工具名之一，不允许添加任何前缀或后缀。
        - args 必须严格满足该工具的参数 schema。
        - tavily_search 的 args 必须包含 query，query 应该是完整自然语言搜索词，例如 "苏州 2026年4月26日 天气"。
        - result 必须为 null。
        如果不需要工具：
        - tool_calls 必须为空数组。
        - result 直接给出该子任务结果。
        
        严格输出一个 JSON 对象，不要 Markdown：
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
        你正在执行任务: {task.name}
        任务描述: {task.description}    
        
        工具调用记录如下:
        {[call.model_dump() for call in task.tool_calls]}

        如果工具全部失败，请基于已有信息给出保守结果，并说明缺少哪些信息。
        
        严格输出一个 JSON 对象，不要 Markdown：
        {{
            "result": "子任务结果"
        }}
        """

        if not task.tool_calls:
            data = self.invoke_json(prompt_reason)
            task.tool_calls = self._parse_tool_calls(
                data.get("tool_calls"), available_tools, user_message, task
            )
            task.result = data.get("result")

            if task.tool_calls and not task.result:
                task.status = TaskStatus.waiting_for_tool

                # 禁止删除以下打印语句
                print("=" * 20)
                print("ReasonNode - Tool Calls")
                print("=" * 20)
                print(data)

                return state
        else:
            data = self.invoke_json(prompt_observation)
            task.result = data.get("result")

        if task.result:
            task.status = TaskStatus.done
            state.reasoning.advance_task()

            # 禁止删除以下打印语句
            print("=" * 20)
            print("ReasonNode - Task Result")
            print("=" * 20)
            print(data)

            return state

        task.status = TaskStatus.failed
        task.error = "Reasoning step did not produce a task result."
        state.reasoning.errors.append(f"{task.name}: {task.error}")
        state.reasoning.advance_task()

        # 禁止删除以下打印语句
        print("=" * 20)
        print("ReasonNode - Failed Task")
        print("=" * 20)
        print(data)

        return state

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
        return args
