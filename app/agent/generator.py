from __future__ import annotations

import json
from typing import Any, Protocol

from app.agent.state import AgentState


class ChatClient(Protocol):
    """
    Minimal interface required from an LLM client.
    """

    def create(self, **kwargs: Any) -> Any:
        ...


class AnswerGenerator:
    """
    Generates a grounded answer from the retrieved context.

    The generator is dependency-injected so unit tests do not
    require a real LLM/API connection.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature

        if client is None:
            from openai import OpenAI

            self.client = OpenAI()
        else:
            self.client = client

    def generate(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Generate a grounded answer and update AgentState.
        """

        query = state.get("query", "").strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        context = state.get(
            "context",
            [],
        )

        prompt = self._build_prompt(
            query=query,
            context=context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a grounded RAG assistant. "
                        "Answer using the provided context. "
                        "Do not invent facts that are not supported "
                        "by the context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
        )

        answer = self._extract_answer(
            response
        )

        return {
            **state,
            "answer": answer,
            "answer_verified": False,
            "finished": False,
        }

    @staticmethod
    def _build_prompt(
        query: str,
        context: list,
    ) -> str:
        """
        Build a grounded generation prompt.
        """

        context_text = AnswerGenerator._format_context(
            context
        )

        return (
            "Question:\n"
            f"{query}\n\n"
            "Retrieved Context:\n"
            f"{context_text}\n\n"
            "Instructions:\n"
            "1. Answer the question directly.\n"
            "2. Use only information supported by the context.\n"
            "3. If the context is insufficient, say so clearly.\n"
            "4. Do not fabricate facts.\n"
        )

    @staticmethod
    def _format_context(
        context: list,
    ) -> str:
        """
        Convert retrieved context into a deterministic string.
        """

        if not context:
            return "No context was retrieved."

        formatted: list[str] = []

        for index, item in enumerate(
            context,
            start=1,
        ):
            if isinstance(item, str):
                text = item

            elif hasattr(item, "page_content"):
                text = str(
                    item.page_content
                )

            elif isinstance(item, dict):
                text = str(
                    item.get(
                        "page_content",
                        item.get(
                            "content",
                            item,
                        ),
                    )
                )

            else:
                text = str(item)

            formatted.append(
                f"[Context {index}]\n{text}"
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _extract_answer(
        response: Any,
    ) -> str:
        """
        Extract text from an OpenAI-compatible response.
        """

        try:
            answer = response.choices[0].message.content

        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                "Invalid LLM response format."
            ) from exc

        if answer is None:
            raise ValueError(
                "LLM returned an empty response."
            )

        answer = str(answer).strip()

        if not answer:
            raise ValueError(
                "LLM returned an empty response."
            )

        return answer
