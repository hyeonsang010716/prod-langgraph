import json
from typing import AsyncIterator

from starlette.responses import StreamingResponse

from app.core.graph.stream.stream_event import StreamEvent, SSEEventType, ErrorEvent


def format_sse(event: StreamEvent) -> str:
    """StreamEvent를 SSE 텍스트 형식으로 변환합니다.

    출력 형식:
        event: <event_type>
        data: <json_payload>

        (SSE 스펙에 따라 빈 줄로 메시지 구분)
    """
    data_json = json.dumps(event.data, ensure_ascii=False, default=str)
    return f"event: {event.event.value}\ndata: {data_json}\n\n"


async def sse_generator(
    event_stream: AsyncIterator[StreamEvent],
) -> AsyncIterator[str]:
    """StreamEvent 스트림을 SSE 텍스트 스트림으로 변환합니다."""
    try:
        async for event in event_stream:
            yield format_sse(event)
    except GeneratorExit:
        return
    except Exception as e:
        error_event = StreamEvent(
            event=SSEEventType.ERROR,
            data=ErrorEvent(
                error=str(e),
                error_type=type(e).__name__,
            ).model_dump(),
        )
        yield format_sse(error_event)


def create_sse_response(
    event_stream: AsyncIterator[StreamEvent],
) -> StreamingResponse:
    """SSE StreamingResponse를 생성합니다.

    Headers:
    - Content-Type: text/event-stream
    - Cache-Control: no-cache (버퍼링 방지)
    - Connection: keep-alive
    - X-Accel-Buffering: no (nginx 버퍼링 비활성화)
    """
    return StreamingResponse(
        content=sse_generator(event_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
