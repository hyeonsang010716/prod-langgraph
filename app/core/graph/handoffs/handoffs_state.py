from langchain.agents import AgentState
from typing_extensions import NotRequired


class HandoffsState(AgentState):
    """멀티 에이전트 Handoffs 그래프 상태

    AgentState 상속: messages (add_messages 리듀서) 내장
    active_agent: 현재 활성 에이전트 이름 (선택)
    """
    active_agent: NotRequired[str]
