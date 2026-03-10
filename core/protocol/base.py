"""协议适配器抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from core.protocol.schemas import CanonicalChatRequest


class ProtocolAdapter(ABC):
    protocol_name: str

    @abstractmethod
    def parse_request(
        self,
        provider: str,
        raw_body: dict[str, Any],
    ) -> CanonicalChatRequest: ...

    @abstractmethod
    def render_non_stream(
        self,
        req: CanonicalChatRequest,
        raw_chunks: list[str],
    ) -> dict[str, Any]: ...

    @abstractmethod
    def render_stream(
        self,
        req: CanonicalChatRequest,
        raw_stream: AsyncIterator[str],
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    def render_error(self, exc: Exception) -> tuple[int, dict[str, Any]]: ...
