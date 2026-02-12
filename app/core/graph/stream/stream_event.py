from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SSEEventType(str, Enum):
    """SSE 이벤트 타입"""
    NODE_START = "node_start"
    NODE_END = "node_end"
    TOKEN = "token"
    GRAPH_START = "graph_start"
    GRAPH_END = "graph_end"
    INTERRUPT = "interrupt"
    ERROR = "error"
    METADATA = "metadata"


class TokenEvent(BaseModel):
    """LLM 토큰 스트리밍 이벤트"""
    content: str
    message_id: Optional[str] = None
    langgraph_node: Optional[str] = None
    langgraph_step: Optional[int] = None
    model_name: Optional[str] = None


class NodeStartEvent(BaseModel):
    """노드 실행 시작 이벤트"""
    node_name: str
    updates: dict = Field(default_factory=dict)


class NodeEndEvent(BaseModel):
    """노드 실행 완료 이벤트"""
    node_name: str
    updates: dict = Field(default_factory=dict)


class InterruptEvent(BaseModel):
    """인터럽트 발생 이벤트"""
    interrupt_info: dict = Field(default_factory=dict)
    execution_time: float = 0.0


class GraphEndEvent(BaseModel):
    """그래프 실행 완료 이벤트"""
    status: str
    answer: str = ""
    execution_time: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0


class ErrorEvent(BaseModel):
    """에러 이벤트"""
    error: str
    error_type: str = "unknown"


class StreamEvent(BaseModel):
    """SSE 이벤트 래퍼"""
    event: SSEEventType
    data: dict = Field(default_factory=dict)
