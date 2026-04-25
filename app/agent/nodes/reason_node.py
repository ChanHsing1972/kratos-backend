import json

from app.agent.nodes.base_node import BaseNode
from app.agent.state.reasoning import Task, TaskStatus


class ReasonNode(BaseNode):

    def __call__(self, state):
        idx = state.reasoning.current_task_index
        task: Task = state.reasoning.tasks[idx]
        tool_calls = task.tool_calls
        tool_results = task.tool_results
        convs = state.conversation.conversations

        self.llm = self.llm.bind_tools(state.tools.available_tools.values())
        task.status = TaskStatus.running

        prompt_reason = f"""        
        你正在执行任务: {task.name}
        任务描述: {task.description}
        
        查询工具之前，先查看过往会话，看有没有什么有用信息:
        {convs}
        
        判断是否需要调用工具来完成任务，如果需要，选择合适的工具并提供必要的参数，result则为空。
        如果不需要调用工具，则直接生成result，tool_calls则为空
        
        输出纯JSON:
        (你的回答必须严格只输出纯JSON字符串，绝对不要添加```json)
        {{          
            "tool_calls": [
                {{
                    "tool_name": "...",
                    "args": {{}},
                    "id": “...”
                }},
                {{
                    "tool_name": "...",
                    "args": {{}},
                    "id": “...”
                }},
            ],
            "result": "...",
        }}
        """

        prompt_observation = f"""
        你正在执行任务: {task.name}
        任务描述: {task.description}    
        
        工具已经调用完成，工具结果如下:
        {tool_results}
        如果工具结果为空则不依赖工具做出回答
        
        输出纯JSON:
        (你的回答必须严格只输出纯JSON字符串，绝对不要添加```json)
        {{
            "result": "...",
        }}
        """
        if not tool_results and not tool_calls:
            result = self.llm.invoke(prompt_reason)

            print("ReasonNode1：")
            print(result)

            data = json.loads(result.content)
            task.tool_calls = data["tool_calls"]
            task.result = data["result"]
        else:
            result = self.llm.invoke(prompt_observation)

            print("ReasonNode2：")
            print(result)

            data = json.loads(result.content)
            task.result = data["result"]

        if task.result:
            task.status = TaskStatus.done

            state.reasoning.current_task_index += 1


        return state
