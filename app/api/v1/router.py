from fastapi import APIRouter

from app.dto.chat_dto import ChatRequest, ChatResponse
from app.service.chat_service import ChatService

router = APIRouter(prefix="/v1", tags=["v1"])

chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """LangGraph 채팅 API"""
    return await chat_service.chat(request)
