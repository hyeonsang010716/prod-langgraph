from typing import AsyncIterator

from app.dto.chat_dto import ChatRequest
from app.core.graph.stream.stream_graph import get_stream_graph
from app.core.graph.stream.stream_event import StreamEvent


class StreamService:
    """SSE 스트리밍 전용 서비스"""

    def __init__(self):
        self.graph = get_stream_graph()

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """SSE 스트리밍으로 채팅 응답을 생성합니다."""
        async for event in self.graph.astream(
            question=request.question,
            session_id=request.session_id,
        ):
            yield event
