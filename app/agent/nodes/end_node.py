from app.agent.nodes.base_node import BaseNode
from app.agent.state.conversation import AskAns
from app.agent.state.session_state import SessionState


class EndNode(BaseNode):

    def __call__(self, state: SessionState):
        human_message = state.conversation.messages[0].content
        ai_message = state.conversation.messages[-1].content

        ask_ans: AskAns = AskAns(user_ask=human_message, ai_ans=ai_message)
        state.conversation.conversations.append(ask_ans)

        state.conversation.messages.clear()

        return state