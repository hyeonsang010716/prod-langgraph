from app.dto.chat_dto import (
    ChatRequest, ChatResponse, DeleteMessagesRequest,
    MessageInfo, CheckpointInfo, CheckpointListResponse,
    PurgeCheckpointsRequest, ResumeRequest,
)
from app.core.graph.example.graph_orchestrator import get_example_graph
from app.util.memorysaver import inspect_all_checkpoints, inspect_single_checkpoint

class ChatService:
    """채팅 서비스"""

    def __init__(self):
        self.graph = get_example_graph()
        

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """LangGraph를 통해 채팅 응답을 생성합니다."""
        result = await self.graph.ainvoke(
            question=request.question,
            session_id=request.session_id,
        )

        if result["status"] == "interrupted":
            return ChatResponse(
                status="interrupted",
                execution_time=round(result["execution_time"], 3),
                interrupt_info=result["interrupt_info"],
            )

        return ChatResponse(
            status="completed",
            answer=result["answer"],
            execution_time=round(result["execution_time"], 3),
            total_tokens=result["total_tokens"],
            total_cost=result["total_cost"],
            history=result["messages"],
        )

    async def resume(self, request: ResumeRequest) -> ChatResponse:
        """interrupt된 그래프를 재개합니다."""
        result = await self.graph.aresume(
            session_id=request.session_id,
            action=request.action,
        )

        if result["status"] == "interrupted":
            return ChatResponse(
                status="interrupted",
                execution_time=round(result["execution_time"], 3),
                interrupt_info=result["interrupt_info"],
            )

        return ChatResponse(
            status="completed",
            answer=result["answer"],
            execution_time=round(result["execution_time"], 3),
            total_tokens=result["total_tokens"],
            total_cost=result["total_cost"],
            history=result["messages"],
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


    async def get_checkpoint_list(self, checkpointer, session_id: str) -> CheckpointListResponse:
        """복원 가능한 체크포인트 목록을 조회합니다."""
        config = {"configurable": {"thread_id": session_id}}

        checkpoints = []
        prev_messages_count = None

        async for cp in checkpointer.alist(config):
            # 부모 그래프만 (서브그래프 제외)
            if cp.config["configurable"].get("checkpoint_ns", "") != "":
                continue

            # loop 소스만 (input 소스는 아직 처리 전 상태)
            if cp.metadata.get("source") != "loop":
                continue

            messages = cp.checkpoint.get("channel_values", {}).get("messages", [])
            messages_count = len(messages)

            # messages 수가 변경된 체크포인트만 (중복 제거)
            if messages_count == prev_messages_count:
                continue
            prev_messages_count = messages_count

            checkpoint_info = CheckpointInfo(
                checkpoint_id=cp.config["configurable"]["checkpoint_id"],
                step=cp.metadata.get("step", 0),
                timestamp=cp.checkpoint.get("ts", ""),
                messages_count=messages_count,
                messages=[
                    MessageInfo(
                        id=msg.id,
                        type=msg.type,
                        content=msg.content[:200],
                    )
                    for msg in messages
                ],
            )
            checkpoints.append(checkpoint_info)

        return CheckpointListResponse(
            session_id=session_id,
            total=len(checkpoints),
            checkpoints=checkpoints,
        )

    async def purge_checkpoints(self, checkpointer, data: PurgeCheckpointsRequest) -> bool:
        """PostgreSQL에서 체크포인트를 물리적으로 전체 삭제합니다."""
        session_id = data.session_id

        # 전체 삭제 - LangGraph 공식 API 사용
        await checkpointer.adelete_thread(session_id)

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
    