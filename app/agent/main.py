import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.agent.state.session_state import SessionState
from app.agent.state.tools import ToolsState

load_dotenv()

def main():
    llm = ChatOpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        temperature=0.5,
    )

    tools = [TavilySearch(max_results=2)]

    available_tools = {tool.name: tool for tool in tools}

    initial_state = SessionState(
        session_id="1",
        user_id="1",
        tools=ToolsState(available_tools=available_tools),
    )

    graph = build_graph(llm)


    while True:
        # 输入问题
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("👋 结束对话")
            break

        # 把用户消息追加到 state.conversation.messages
        initial_state.conversation.messages.append(
            HumanMessage(content=user_input)
        )

        print("当前状态:", initial_state)

        # 🔥 把当前 state 丢进图执行
        initial_state = graph.invoke(initial_state)
        initial_state = SessionState(**initial_state)

        print("当前状态:", initial_state)

        # 输出AI回答
        print("AI:", initial_state.result.response)

if __name__ == "__main__":
    main()


