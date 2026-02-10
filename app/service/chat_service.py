from app.dto.chat_dto import ChatRequest, ChatResponse, DeleteMessagesRequest
from app.core.graph.example.graph_orchestrator import get_example_graph
from app.util.memorysaver import inspect_all_checkpoints, inspect_single_checkpoint

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
        
        
    async def info(self, checkpointer):
        """LangGraph MemorySaver 정보 추출"""

        # # 모든 체크포인트 조회
        # await inspect_all_checkpoints(checkpointer, config=None, limit=20)
        
        # 특정 thread만 조회
        config = {"configurable": {"thread_id": "u1"}}
        await inspect_all_checkpoints(checkpointer, config=config, limit=10)
        
        # # 단일 체크포인트 상세 조회
        # config = {"configurable": {"thread_id": "t1", "checkpoint_id": "abc123"}}
        # await inspect_single_checkpoint(checkpointer, config)

        return True


    async def delete_messages(self, request: DeleteMessagesRequest) -> dict:
        """메시지를 삭제합니다. message_ids가 없으면 전체 삭제."""
        if request.message_ids:
            return await self.graph.adelete_messages(
                session_id=request.session_id,
                message_ids=request.message_ids,
            )
        else:
            return await self.graph.aclear_history(
                session_id=request.session_id,
            )
    