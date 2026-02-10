from fastapi import APIRouter, Request

from app.dto.chat_dto import ChatRequest, ChatResponse, DeleteMessagesRequest
from app.service.chat_service import ChatService

router = APIRouter(prefix="/v1", tags=["v1"])

chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """LangGraph 채팅 API"""
    return await chat_service.chat(request)


@router.post("/info")
async def info(request: Request):
    """LangGraph Checkpoint 정보 조회 API"""
    checkpointer = request.app.state.checkpointer
    return await chat_service.info(checkpointer)


@router.delete("/chat/messages")
async def delete_messages(request: DeleteMessagesRequest):
    """메시지 삭제 API (message_ids 없으면 전체 삭제)"""
    return await chat_service.delete_messages(request)