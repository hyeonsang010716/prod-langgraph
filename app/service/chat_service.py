from app.dto.chat_dto import ChatRequest, ChatResponse
from app.core.graph.example.graph_orchestrator import get_example_graph


class ChatService:
    """채팅 서비스"""

    def __init__(self):
        self.graph = get_example_graph()
        

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """LangGraph를 통해 채팅 응답을 생성합니다."""

        answer, execution_time, total_tokens, total_cost, history = await self.graph.ainvoke(
            question=request.question,
            session_id=request.session_id,
        )

        return ChatResponse(
            answer=answer,
            execution_time=round(execution_time, 3),
            total_tokens=total_tokens,
            total_cost=total_cost,
            history=history
        )
