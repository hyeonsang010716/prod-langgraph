from app.core.graph.handoffs.handoffs_graph import get_handoffs_graph


class HandoffsService:
    """멀티 에이전트 Handoffs 서비스"""

    def __init__(self):
        self.graph = get_handoffs_graph()

    async def chat(self, question: str, session_id: str) -> dict:
        """Handoffs 에이전트에게 질문합니다."""
        return await self.graph.ainvoke(
            question=question,
            session_id=session_id,
        )

    async def resume(self, session_id: str, action: str) -> dict:
        """interrupt된 Handoffs 그래프를 재개합니다."""
        return await self.graph.aresume(
            session_id=session_id,
            action=action,
        )
