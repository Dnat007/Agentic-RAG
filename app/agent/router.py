from app.agent.state import AgentState


class AgentRouter:
    """
    Lightweight routing gate.

    The router does NOT try to understand every possible user query.

    Semantic understanding is handled by the Planner.
    """

    def route(self, state: AgentState) -> AgentState:
        query = state.get("query", "").strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        # Planner should decide the actual intent/action.
        state.setdefault("intent", "unknown")
        state.setdefault("next_action", "planner")

        return state