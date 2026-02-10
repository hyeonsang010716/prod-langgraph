from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """채팅 요청 DTO"""
    question: str = Field(..., description="사용자 질문", min_length=1)
    session_id: str = Field(..., description="세션 ID")


class ChatResponse(BaseModel):
    """채팅 응답 DTO"""
    answer: str = Field(..., description="AI 응답")
    execution_time: float = Field(..., description="실행 시간 (초)")
    total_tokens: int = Field(..., description="총 토큰 수")
    total_cost: float = Field(..., description="총 비용 (USD)")
    history: List = Field(default_factory=[], description="히스토리")


class DeleteMessagesRequest(BaseModel):
    """메시지 삭제 요청 DTO"""
    session_id: str = Field(..., description="세션 ID")
    message_ids: Optional[List[str]] = Field(None, description="삭제할 메시지 ID 목록 (없으면 전체 삭제)")
