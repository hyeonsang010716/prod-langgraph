from typing import Optional, Any, AsyncIterator
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from functools import lru_cache

from app.core.graph.stream.stream_state import StreamGraphState
from app.core.graph.stream.stream_processor import StreamProcessor
from app.core.graph.stream.stream_event import StreamEvent
from app.core.llm_manager import ModelName, get_llm_manager


SYSTEM_PROMPT = """당신은 유저의 질문에 친절하고 명확하게 답변하는 AI 어시스턴트입니다.
이전 대화 히스토리를 참고하여 자연스럽게 대화를 이어가세요."""


class StreamGraphOrchestrator:
    """SSE 스트리밍 전용 그래프 오케스트레이터

    실제 LLM을 호출하여 토큰 단위로 SSE 스트리밍합니다.
    독립적인 그래프로, example 그래프와 분리되어 동작합니다.

    그래프 플로우:
        START → GenerateResponse (LLM 호출) → END
    """

    def __init__(self):
        self._graph: Optional[Any] = None
        self._llm_manager = get_llm_manager()

    async def initialize(self, store, checkpointer) -> None:
        """그래프를 빌드하고 컴파일합니다."""
        await self._build_graph(store, checkpointer)

    # ===== 그래프 노드 =====

    async def _generate_response(self, state: StreamGraphState) -> dict:
        """LLM을 호출하여 응답을 생성합니다.

        messages 히스토리 전체를 LLM에 전달하여 맥락을 유지합니다.
        stream_mode="messages" 사용 시, 이 노드의 LLM 호출이
        토큰 단위로 자동 스트리밍됩니다.
        """
        model = self._llm_manager.get_model(ModelName.GPT_4O_MINI)

        # 시스템 프롬프트 + 대화 히스토리
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = await model.ainvoke(messages)
        return {"messages": [response]}

    async def _build_graph(self, store, checkpointer) -> None:
        """LangGraph를 구성합니다."""
        builder = StateGraph(StreamGraphState)

        builder.add_node("GenerateResponse", self._generate_response)

        builder.add_edge(START, "GenerateResponse")
        builder.add_edge("GenerateResponse", END)

        self._graph = builder.compile(
            store=store,
            checkpointer=checkpointer,
        )

    # ===== 스트리밍 API =====

    async def astream(self, question: str, session_id: str) -> AsyncIterator[StreamEvent]:
        """사용자 질문을 SSE 이벤트로 스트리밍합니다."""
        input_data = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": session_id}}

        processor = StreamProcessor(self._graph)
        async for event in processor.astream_events(input_data, config):
            yield event


@lru_cache(maxsize=1)
def get_stream_graph() -> StreamGraphOrchestrator:
    """StreamGraphOrchestrator 싱글톤 인스턴스를 반환합니다."""
    return StreamGraphOrchestrator()
