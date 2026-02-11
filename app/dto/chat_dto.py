from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """채팅 요청 DTO"""
    question: str = Field(..., description="사용자 질문", min_length=1)
    session_id: str = Field(..., description="세션 ID")


class ChatResponse(BaseModel):
    """채팅 응답 DTO"""
    status: str = Field("completed", description="상태 (completed/interrupted)")
    answer: str = Field("", description="AI 응답")
    execution_time: float = Field(0, description="실행 시간 (초)")
    total_tokens: int = Field(0, description="총 토큰 수")
    total_cost: float = Field(0, description="총 비용 (USD)")
    history: List = Field(default_factory=list, description="히스토리")
    interrupt_info: Optional[dict] = Field(None, description="인터럽트 정보")


class ResumeRequest(BaseModel):
    """Human-in-the-loop 재개 요청 DTO"""
    session_id: str = Field(..., description="세션 ID")
    action: str = Field(..., description="approve: 승인 / 문자열: 수정된 분석 내용")


class DeleteMessagesRequest(BaseModel):
    """메시지 삭제 요청 DTO"""
    session_id: str = Field(..., description="세션 ID")
    message_ids: Optional[List[str]] = Field(None, description="삭제할 메시지 ID 목록 (없으면 전체 삭제)")


class MessageInfo(BaseModel):
    """메시지 정보"""
    id: str = Field(..., description="메시지 ID")
    type: str = Field(..., description="메시지 타입 (human/ai)")
    content: str = Field(..., description="메시지 내용")


class CheckpointInfo(BaseModel):
    """체크포인트 정보"""
    checkpoint_id: str = Field(..., description="체크포인트 ID")
    step: int = Field(..., description="실행 스텝")
    timestamp: str = Field(..., description="생성 시각")
    messages_count: int = Field(..., description="메시지 수")
    messages: List[MessageInfo] = Field(default_factory=list, description="메시지 목록")


class CheckpointListResponse(BaseModel):
    """체크포인트 목록 응답 DTO"""
    session_id: str = Field(..., description="세션 ID")
    total: int = Field(..., description="총 체크포인트 수")
    checkpoints: List[CheckpointInfo] = Field(default_factory=list, description="체크포인트 목록")


class PurgeCheckpointsRequest(BaseModel):
    """체크포인트 물리 삭제 요청 DTO"""
    session_id: str = Field(..., description="세션 ID")
