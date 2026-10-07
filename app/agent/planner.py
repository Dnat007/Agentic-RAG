import json

from openai import OpenAI

from app.agent.state import AgentState
from app.config.settings import get_settings


class AgentPlanner:
    """
    LLM-based planner for the LangGraph agent.

    The planner decides:
    - intent
    - required capabilities
    - execution plan
    - next action

    It does not execute tools itself.
    """

    VALID_INTENTS = {
        "document",
        "external",
        "computation",
        "structured_data",
        "multi_source",
        "direct_response",
        "unknown",
    }

    VALID_ACTIONS = {
        "rag",
        "web",
        "calculator",
        "sql",
        "multi_source",
        "direct_response",
    }

    def __init__(self, client=None):
        settings = get_settings()

        self.client = client or OpenAI()
        self.model = settings.llm_model

    def plan(self, state: AgentState) -> AgentState:
        query = state.get("query", "").strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        system_prompt = """
You are the planning component of a production-grade Agentic RAG system.

Your job is to understand the user's request and select the
appropriate capability.

Available capabilities:

1. document
   - Private/indexed documents
   - Company PDFs
   - Internal knowledge

2. external
   - Current/latest information
   - Web or external APIs

3. computation
   - Mathematical calculations
   - Numerical reasoning
   - Python/calculator tasks

4. structured_data
   - SQL/database queries
   - Structured business data

5. multi_source
   - Requires multiple sources
   - Example: private document + latest web information

6. direct_response
   - Can be answered directly without retrieval or tools

Important:
- Do NOT hard-code individual mathematical concepts.
- Do NOT classify based on simple keyword matching.
- Understand the complete semantic meaning of the query.
- Current/latest/today information generally requires external sources.
- Private company/document information should use document retrieval.
- If multiple sources are required, use multi_source.

Return ONLY valid JSON:

{
    "intent": "document | external | computation | structured_data | multi_source | direct_response | unknown",
    "capabilities": ["..."],
    "plan": ["step 1", "step 2"],
    "next_action": "rag | web | calculator | sql | multi_source | direct_response"
}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Planner returned an empty response.")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Planner returned invalid JSON."
            ) from exc

        intent = result.get("intent", "unknown")
        capabilities = result.get("capabilities", [])
        plan = result.get("plan", [])
        next_action = result.get("next_action")

        if intent not in self.VALID_INTENTS:
            intent = "unknown"

        if not isinstance(capabilities, list):
            capabilities = []

        if not isinstance(plan, list):
            plan = []

        if next_action not in self.VALID_ACTIONS:
            next_action = "direct_response"

        # LangGraph state update
        state["intent"] = intent
        state["plan"] = plan
        state["next_action"] = next_action

        # Store capabilities in the plan state if useful
        state["plan"] = [
            *capabilities,
            *plan,
        ]

        return state
