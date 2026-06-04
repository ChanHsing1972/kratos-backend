"""本地命令行调试入口。

生产 API 使用 `app.services.agent_chat`；本文件只用于开发时在终端里快速跑通
Agent 节点链路和工具注册。
"""

from langchain_core.messages import HumanMessage

from app.agent.llm import get_agent_llm
from app.agent.runner import AgentRunner, build_agent_nodes
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolsState
from app.agent.tools import load_tools


def main():
    """启动一个简单的交互式 Agent 会话。"""

    available_tools = load_tools()

    initial_state = SessionState(
        session_id="1",
        user_id="1",
        tools=ToolsState(available_tools=available_tools),
    )

    runner = AgentRunner(build_agent_nodes(get_agent_llm()))

    while True:
        # 输入问题
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("👋 结束对话")
            break

        # 把用户消息追加到 state.conversation.messages
        initial_state.conversation.messages.append(HumanMessage(content=user_input))

        print("=" * 20)
        print("当前状态")
        print("=" * 20)
        print(initial_state)

        initial_state = runner.run(initial_state)

        print("=" * 20)
        print("当前状态")
        print("=" * 20)
        print(initial_state)

        # 输出AI回答
        print("AI:", initial_state.result.response)


if __name__ == "__main__":
    main()
