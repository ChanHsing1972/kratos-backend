from app.agent.nodes.base_node import BaseNode
from app.agent.state.conversation import AskAns
from app.agent.state.session_state import SessionState


class EndNode(BaseNode):

    def __call__(self, state: SessionState):
        if state.conversation.messages:
            human_message = state.conversation.messages[0].content
            ai_message = state.conversation.messages[-1].content
            ask_ans = AskAns(user_ask=human_message, ai_ans=ai_message)
            state.conversation.conversations.append(ask_ans)

        max_conversations = state.conversation.max_conversations
        if len(state.conversation.conversations) > max_conversations:
            overflow = state.conversation.conversations[:-max_conversations]
            state.conversation.conversations = state.conversation.conversations[-max_conversations:]
            summary = "；".join(
                f"用户:{item.user_ask} AI:{item.ai_ans}" for item in overflow
            )
            if summary:
                state.conversation.summaries.append(summary)

        state.conversation.messages.clear()
        state.conversation.first_ai_message = None
        state.reasoning.reset_tasks()
        state.reasoning.reflection = None
        state.reasoning.errors = []
        state.tools.history = []
        state.result.reset_runtime_for_new_turn()

        return state
