from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class StreamGraphState(TypedDict):
    """스트리밍 그래프 상태 정의

    messages: 대화 히스토리 (add_messages 리듀서로 자동 관리)
    """
    messages: Annotated[List, add_messages]
