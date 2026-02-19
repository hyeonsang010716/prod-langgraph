from typing import AsyncIterator

from poc.graph import SupervisorGraphOrchestrator
from poc.registry import AgentRegistry


class PocService:
    """Supervisor Multi-Agent 서비스 레이어

    상담 채팅 + 에이전트/도구 관리 API를 제공합니다.
    """

    def __init__(
        self, orchestrator: SupervisorGraphOrchestrator, registry: AgentRegistry
    ):
        self._graph = orchestrator
        self._registry = registry

    # ── Chat ──

    async def chat(self, question: str, session_id: str) -> dict:
        return await self._graph.ainvoke(question, session_id)

    async def resume(self, session_id: str, message: str) -> dict:
        return await self._graph.aresume(session_id, message)

    # ── Streaming Chat ──

    async def stream_chat(
        self, question: str, session_id: str
    ) -> AsyncIterator[dict]:
        async for event in self._graph.astream_invoke(question, session_id):
            yield event

    async def stream_resume(
        self, session_id: str, message: str
    ) -> AsyncIterator[dict]:
        async for event in self._graph.astream_resume(session_id, message):
            yield event

    # ── Agent Management ──

    def add_agent(self, key: str, description: str, prompt: str) -> dict:
        self._registry.register(key, description, prompt)
        return {"status": "added", "agent": key}

    def remove_agent(self, key: str) -> dict:
        if self._registry.unregister(key):
            return {"status": "removed", "agent": key}
        return {"status": "error", "message": f"'{key}' 에이전트를 찾을 수 없습니다."}

    def list_agents(self) -> list[dict]:
        return [
            {
                "key": k,
                "description": v.description,
                "tools": [t.name for t in v.tools],
            }
            for k, v in self._registry.list_all().items()
        ]

    # ── Tool Management ──

    def add_tool(
        self, agent_key: str, name: str, description: str, mock_response: str
    ) -> dict:
        try:
            self._registry.add_tool(agent_key, name, description, mock_response)
            return {"status": "added", "agent": agent_key, "tool": name}
        except KeyError as e:
            return {"status": "error", "message": str(e)}

    def remove_tool(self, agent_key: str, tool_name: str) -> dict:
        if self._registry.remove_tool(agent_key, tool_name):
            return {"status": "removed", "agent": agent_key, "tool": tool_name}
        return {"status": "error", "message": "도구를 찾을 수 없습니다."}
