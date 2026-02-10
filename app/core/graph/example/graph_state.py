from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from langchain.messages import HumanMessage
from langgraph.graph.message import add_messages


class SubGraphState(TypedDict):
    """서브 그래프 상태 정의"""
    question: HumanMessage                      # 부모로부터 전달받는 질문
    analysis: str                               # 질문 분석 결과
    answer: str                                 # 서브 그래프 최종 응답 → 부모로 전달


class GraphState(TypedDict):
    """LangGraph 상태 정의"""
    messages: Annotated[List, add_messages]     # 메시지 관리
    question: HumanMessage                      # 최신 질문
    documents: List[Document]                   # Retriever 문서
    answer: str                                 # 최종 결과