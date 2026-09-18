"""LangChain chat model backed by NeoSyntropy."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from neosyntropy.providers.neosyntropy import NeoSyntropyProvider

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install LangGraph support with: pip install 'neosyntropy[langgraph]'"
    ) from exc


class NeoSyntropyChatModel(BaseChatModel):
    provider: Any
    model_name: str

    def __init__(self, api_key: str, model: str, **kwargs: Any) -> None:
        super().__init__(provider=NeoSyntropyProvider(api_key, model, **kwargs), model_name=model)

    @property
    def _llm_type(self) -> str:
        return "neosyntropy-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self.provider.chat(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await self.provider.achat(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.content))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        from langchain_core.messages import AIMessageChunk

        result = self.provider.chat(messages, **kwargs)
        yield ChatGeneration(message=AIMessageChunk(content=result.content))
