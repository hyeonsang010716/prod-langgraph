import time
from typing import AsyncIterator, Any

from langchain_core.messages import AIMessageChunk
from langchain_community.callbacks import get_openai_callback

from app.core.graph.stream.stream_event import (
    StreamEvent, SSEEventType,
    TokenEvent, NodeEndEvent,
    InterruptEvent, GraphEndEvent, ErrorEvent,
)


class StreamProcessor:
    """LangGraph astream() 출력을 SSE StreamEvent로 변환하는 프로세서.

    그래프에 독립적으로 설계되어 어떤 LangGraph 그래프에서든 재사용 가능합니다.
    stream_mode=["messages", "updates"]를 사용하여:
      - "messages": LLM 토큰 스트리밍 (AIMessageChunk → TOKEN 이벤트)
      - "updates": 노드 상태 변경 및 __interrupt__ 감지
    """

    def __init__(self, graph: Any):
        self._graph = graph

    async def astream_events(
        self,
        input_data: Any,
        config: dict,
    ) -> AsyncIterator[StreamEvent]:
        """그래프 실행을 SSE 이벤트 스트림으로 변환합니다."""
        start_time = time.perf_counter()
        total_tokens = 0
        total_cost = 0.0
        final_answer = ""
        status = "completed"

        # graph_start 이벤트
        yield StreamEvent(
            event=SSEEventType.GRAPH_START,
            data={"timestamp": time.time()},
        )

        try:
            with get_openai_callback() as cb:
                async for chunk in self._graph.astream(
                    input_data,
                    config,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                ):
                    # subgraphs=True: (namespace_tuple, mode, payload)
                    # subgraphs=False: (mode, payload)
                    if len(chunk) == 3:
                        _ns, mode, payload = chunk
                    else:
                        mode, payload = chunk

                    if mode == "messages":
                        event = self._process_message_event(payload)
                        if event:
                            yield event

                    elif mode == "updates":
                        events = self._process_update_event(
                            payload, start_time
                        )
                        for event in events:
                            if event.event == SSEEventType.INTERRUPT:
                                status = "interrupted"
                            yield event

                total_tokens = cb.total_tokens
                total_cost = cb.total_cost

        except Exception as e:
            yield StreamEvent(
                event=SSEEventType.ERROR,
                data=ErrorEvent(
                    error=str(e),
                    error_type=type(e).__name__,
                ).model_dump(),
            )
            return

        execution_time = time.perf_counter() - start_time

        # 스트림 종료 후 인터럽트 상태 최종 확인
        state = await self._graph.aget_state(config)
        if state.next:
            status = "interrupted"
            interrupt_info = self._extract_interrupt_from_state(state)
            yield StreamEvent(
                event=SSEEventType.INTERRUPT,
                data=InterruptEvent(
                    interrupt_info=interrupt_info,
                    execution_time=round(execution_time, 3),
                ).model_dump(),
            )
        else:
            # 완료된 경우 최종 answer 추출
            values = state.values
            if "answer" in values and values["answer"]:
                answer = values["answer"]
                final_answer = answer.content if hasattr(answer, "content") else str(answer)
            elif "messages" in values and values["messages"]:
                # messages 기반 그래프: 마지막 AI 메시지에서 추출
                last_msg = values["messages"][-1]
                if hasattr(last_msg, "content"):
                    final_answer = last_msg.content

        # graph_end 이벤트
        yield StreamEvent(
            event=SSEEventType.GRAPH_END,
            data=GraphEndEvent(
                status=status,
                answer=final_answer,
                execution_time=round(execution_time, 3),
                total_tokens=total_tokens,
                total_cost=round(total_cost, 6),
            ).model_dump(),
        )

    def _process_message_event(self, payload: tuple) -> StreamEvent | None:
        """messages 모드 이벤트 처리: (BaseMessage, metadata) 튜플."""
        message, metadata = payload

        # AIMessageChunk만 토큰 이벤트로 변환 (완성된 AIMessage는 무시)
        if isinstance(message, AIMessageChunk) and message.content:
            content = message.content if isinstance(message.content, str) else str(message.content)
            return StreamEvent(
                event=SSEEventType.TOKEN,
                data=TokenEvent(
                    content=content,
                    message_id=message.id,
                    langgraph_node=metadata.get("langgraph_node"),
                    langgraph_step=metadata.get("langgraph_step"),
                    model_name=metadata.get("ls_model_name"),
                ).model_dump(),
            )
        return None

    def _process_update_event(
        self, payload: dict, start_time: float
    ) -> list[StreamEvent]:
        """updates 모드 이벤트 처리: {node_name: state_update} 또는 {__interrupt__: [...]}."""
        events = []

        if "__interrupt__" in payload:
            # 인터럽트 감지
            for interrupt_data in payload["__interrupt__"]:
                value = (
                    interrupt_data.value
                    if hasattr(interrupt_data, "value")
                    else interrupt_data
                )
                events.append(StreamEvent(
                    event=SSEEventType.INTERRUPT,
                    data=InterruptEvent(
                        interrupt_info=value if isinstance(value, dict) else {"value": str(value)},
                        execution_time=round(time.perf_counter() - start_time, 3),
                    ).model_dump(),
                ))
        else:
            # 노드 완료 이벤트
            for node_name, updates in payload.items():
                events.append(StreamEvent(
                    event=SSEEventType.NODE_END,
                    data=NodeEndEvent(
                        node_name=node_name,
                        updates=updates if isinstance(updates, dict) else {},
                    ).model_dump(),
                ))

        return events

    @staticmethod
    def _extract_interrupt_from_state(state) -> dict:
        """그래프 상태에서 인터럽트 정보를 추출합니다."""
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
        return {}
