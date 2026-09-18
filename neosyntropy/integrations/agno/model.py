"""Agno model backed by the NeoSyntropy chat gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from neosyntropy.providers.neosyntropy import NeoSyntropyProvider

try:
    from agno.models.base import Model
    from agno.models.message import Message
    from agno.models.response import ModelResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install Agno support with: pip install 'neosyntropy[agno]'") from exc


class NeoSyntropyModel(Model):
    """Agno model that delegates inference to NeoSyntropy."""

    def __init__(
        self,
        id: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or id
        if not api_key:
            raise ValueError("api_key is required")
        if not resolved_model:
            raise ValueError("model is required")
        self._provider = NeoSyntropyProvider(
            api_key, resolved_model, **kwargs.pop("provider_kwargs", {})
        )
        super().__init__(id=resolved_model, provider="neosyntropy", **kwargs)

    def invoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        result = self._provider.chat(messages, **kwargs)
        return ModelResponse(content=result.content)

    async def ainvoke(self, messages: list[Message], **kwargs: Any) -> ModelResponse:
        result = await self._provider.achat(messages, **kwargs)
        return ModelResponse(content=result.content)

    def invoke_stream(self, messages: list[Message], **kwargs: Any) -> Iterator[ModelResponse]:
        result = self._provider.chat(messages, stream=True, **kwargs)
        yield ModelResponse(content=result.content)

    async def ainvoke_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[ModelResponse]:
        result = await self._provider.achat(messages, stream=True, **kwargs)
        yield ModelResponse(content=result.content)

    def _parse_provider_response(self, response: Any, **_: Any) -> ModelResponse:
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(content=str(response))

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._parse_provider_response(response)
