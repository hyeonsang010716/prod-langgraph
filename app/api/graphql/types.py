import strawberry
from typing import Optional


@strawberry.type
class ChatResponseType:
    """채팅 응답 타입"""
    status: str
    answer: str = ""
    execution_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    interrupt_info: Optional[strawberry.scalars.JSON] = None


@strawberry.type
class MessageInfoType:
    """메시지 정보 타입"""
    id: str
    type: str
    content: str


@strawberry.type
class CheckpointInfoType:
    """체크포인트 정보 타입"""
    checkpoint_id: str
    step: int
    timestamp: str
    messages_count: int
    messages: list[MessageInfoType]


@strawberry.type
class CheckpointListType:
    """체크포인트 목록 타입"""
    session_id: str
    total: int
    checkpoints: list[CheckpointInfoType]


@strawberry.type
class DeleteResultType:
    """삭제 결과 타입"""
    deleted: int
    remaining: int = 0
