from langgraph.graph import MessagesState


class SupervisorState(MessagesState):
    """Supervisor Multi-Agent 그래프 상태

    MessagesState 상속: messages (add_messages 리듀서) 내장
    active_agent: 현재 활성 에이전트 이름 (빈 문자열 = 미할당)
    """

    active_agent: str
