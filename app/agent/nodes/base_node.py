from app.agent.state.session_state import SessionState


class BaseNode:
    def __init__(self, llm=None):
        self.llm = llm

    def __call__(self, state: SessionState) -> SessionState:
        raise NotImplementedError("Subclasses must implement this method")

