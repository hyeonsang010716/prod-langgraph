import strawberry
from typing import Optional

from app.api.graphql.types import (
    ChatResponseType, CheckpointListType, CheckpointInfoType,
    MessageInfoType, DeleteResultType,
)
from app.dto.chat_dto import (
    ChatRequest, ResumeRequest, DeleteMessagesRequest,
)
from app.service.chat_service import ChatService

chat_service = ChatService()


@strawberry.type
class Query:

    @strawberry.field(description="체크포인트 목록 조회")
    async def checkpoints(self, info: strawberry.types.Info, session_id: str) -> CheckpointListType:
        checkpointer = info.context["request"].app.state.checkpointer
        result = await chat_service.get_checkpoint_list(checkpointer, session_id)

        return CheckpointListType(
            session_id=result.session_id,
            total=result.total,
            checkpoints=[
                CheckpointInfoType(
                    checkpoint_id=cp.checkpoint_id,
                    step=cp.step,
                    timestamp=cp.timestamp,
                    messages_count=cp.messages_count,
                    messages=[
                        MessageInfoType(id=m.id, type=m.type, content=m.content)
                        for m in cp.messages
                    ],
                )
                for cp in result.checkpoints
            ],
        )


@strawberry.type
class Mutation:

    @strawberry.mutation(description="채팅 메시지 전송")
    async def chat(self, question: str, session_id: str) -> ChatResponseType:
        request = ChatRequest(question=question, session_id=session_id)
        result = await chat_service.chat(request)

        return ChatResponseType(
            status=result.status,
            answer=result.answer,
            execution_time=result.execution_time,
            total_tokens=result.total_tokens,
            total_cost=result.total_cost,
            interrupt_info=result.interrupt_info,
        )

    @strawberry.mutation(description="인터럽트된 그래프 재개")
    async def resume(self, session_id: str, action: str) -> ChatResponseType:
        request = ResumeRequest(session_id=session_id, action=action)
        result = await chat_service.resume(request)

        return ChatResponseType(
            status=result.status,
            answer=result.answer,
            execution_time=result.execution_time,
            total_tokens=result.total_tokens,
            total_cost=result.total_cost,
            interrupt_info=result.interrupt_info,
        )

    @strawberry.mutation(description="메시지 삭제 (message_ids 없으면 전체 삭제)")
    async def delete_messages(
        self,
        session_id: str,
        message_ids: Optional[list[str]] = None,
    ) -> DeleteResultType:
        request = DeleteMessagesRequest(session_id=session_id, message_ids=message_ids)
        result = await chat_service.delete_messages(request)

        return DeleteResultType(
            deleted=result["deleted"],
            remaining=result.get("remaining", 0),
        )
