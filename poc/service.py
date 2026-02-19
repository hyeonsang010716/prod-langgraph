from poc.graph import SupervisorGraphOrchestrator
from poc.registry import AgentRegistry


class PocService:
    """Supervisor Multi-Agent 서비스 레이어

    상담 채팅 + 에이전트 관리 API를 제공합니다.
    """

    def __init__(
        self, orchestrator: SupervisorGraphOrchestrator, registry: AgentRegistry
    ):
        self._graph = orchestrator
        self._registry = registry

    # ── Chat ──

    async def chat(self, question: str, session_id: str) -> dict:
        """새 대화를 시작합니다."""
        return await self._graph.ainvoke(question, session_id)

    async def resume(self, session_id: str, message: str) -> dict:
        """interrupt된 대화를 재개합니다."""
        return await self._graph.aresume(session_id, message)

    # ── Agent Management ──

    def add_agent(self, key: str, description: str, prompt: str) -> dict:
        """에이전트를 추가합니다."""
        self._registry.register(key, description, prompt)
        return {"status": "added", "agent": key}

    def remove_agent(self, key: str) -> dict:
        """에이전트를 삭제합니다."""
        if self._registry.unregister(key):
            return {"status": "removed", "agent": key}
        return {"status": "error", "message": f"'{key}' 에이전트를 찾을 수 없습니다."}

    def list_agents(self) -> list[dict]:
        """등록된 에이전트 목록을 반환합니다."""
        return [
            {"key": k, "description": v.description}
            for k, v in self._registry.list_all().items()
        ]
