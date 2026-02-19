import hashlib
import json
import time
from typing import Optional, Any, AsyncIterator

from langchain.agents import create_agent
from langchain.messages import SystemMessage, AIMessage, HumanMessage, AIMessageChunk
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from poc.state import SupervisorState
from poc.registry import AgentRegistry, AgentConfig, ToolConfig
from app.core.llm_manager import ModelName, get_llm_manager


SUPERVISOR_PROMPT_TEMPLATE = """\
당신은 고객 상담 Supervisor입니다.
고객의 메시지를 분석하여 가장 적합한 전문 에이전트로 라우팅합니다.

사용 가능한 에이전트:
{agent_descriptions}
- end: 고객이 상담 종료를 원할 때

라우팅 규칙:
1. 고객의 최신 메시지를 기반으로 의도를 판단합니다.
2. 메시지의 주제가 명확하면 해당 에이전트로 라우팅합니다.
3. 고객이 "종료", "그만", "끝", "감사합니다" 등 종료 의사를 표현하면 end로 라우팅합니다.
4. 이전 대화 맥락을 고려하여 자연스럽게 라우팅합니다."""


# ===== Tool Factory =====


def _build_tools(tool_configs: list[ToolConfig]) -> list[StructuredTool]:
    """ToolConfig 목록을 LangChain StructuredTool로 변환합니다."""
    tools = []
    for tc in tool_configs:

        def _make_handler(response_template: str):
            def handler(query: str) -> str:
                if "{query}" in response_template:
                    return response_template.format(query=query)
                return response_template

            return handler

        tool = StructuredTool.from_function(
            func=_make_handler(tc.mock_response),
            name=tc.name,
            description=tc.description,
        )
        tools.append(tool)
    return tools


# ===== Agent Cache =====


def _config_hash(config: AgentConfig) -> str:
    """AgentConfig의 해시 (프롬프트+도구 변경 감지용)."""
    data = {
        "prompt": config.prompt,
        "tools": [(t.name, t.description, t.mock_response) for t in config.tools],
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class _AgentCache:
    """create_react_agent 서브그래프 캐시

    에이전트 설정(프롬프트, 도구)이 변경될 때만 서브그래프를 재생성합니다.
    동일 설정이면 캐시된 서브그래프를 재사용합니다.
    """

    def __init__(self):
        self._graphs: dict[str, Any] = {}
        self._hashes: dict[str, str] = {}

    def get_or_create(self, config: AgentConfig, model: Any) -> Any:
        h = _config_hash(config)
        if self._hashes.get(config.key) != h:
            tools = _build_tools(config.tools)
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=config.prompt,
                name=config.key,
            )
            self._graphs[config.key] = agent
            self._hashes[config.key] = h
        return self._graphs[config.key]

    def invalidate(self, key: str) -> None:
        self._graphs.pop(key, None)
        self._hashes.pop(key, None)


# ===== Graph Orchestrator =====


class SupervisorGraphOrchestrator:
    """Dynamic Supervisor 기반 멀티 에이전트 오케스트레이터

    - 부모 그래프: 고정 구조 (supervisor ↔ agent)
    - 에이전트 노드: create_react_agent 서브그래프를 동적 생성/캐싱
    - 에이전트/도구 추가·삭제 시 그래프 재컴파일 불필요

    그래프 플로우:
        START → supervisor → agent(서브그래프) → (interrupt) → supervisor → ... → END

    서브그래프 내부:
        agent(LLM) → tools → agent(LLM) → ... → 최종 응답
    """

    def __init__(self, registry: AgentRegistry):
        self._graph: Optional[Any] = None
        self._registry = registry
        self._llm_manager = get_llm_manager()

    async def initialize(self, store, checkpointer) -> None:
        """그래프를 빌드하고 컴파일합니다."""
        model = self._llm_manager.get_model(ModelName.GPT_4O_MINI)
        registry = self._registry
        agent_cache = _AgentCache()

        # Supervisor 스키마/프롬프트 캐시
        _sv_cache: dict = {"version": -1, "llm": None, "prompt": ""}

        def _refresh_supervisor_cache() -> None:
            if registry.version == _sv_cache["version"]:
                return
            schema = registry.build_route_schema()
            _sv_cache["llm"] = model.with_structured_output(schema)
            _sv_cache["prompt"] = SUPERVISOR_PROMPT_TEMPLATE.format(
                agent_descriptions=registry.get_descriptions()
            )
            _sv_cache["version"] = registry.version

        # ── Supervisor 노드 ──
        async def supervisor_node(state: SupervisorState) -> Command:
            _refresh_supervisor_cache()

            messages = [SystemMessage(content=_sv_cache["prompt"])] + state["messages"]
            decision = await _sv_cache["llm"].ainvoke(messages)
            next_agent = decision.next.value

            if next_agent == "end":
                farewell = AIMessage(
                    content="상담을 종료합니다. 이용해 주셔서 감사합니다."
                )
                return Command(
                    goto=END,
                    update={"messages": [farewell], "active_agent": "end"},
                )

            return Command(
                goto="agent",
                update={"active_agent": next_agent},
            )

        # ── 범용 에이전트 노드 (create_react_agent 서브그래프 실행) ──
        async def agent_node(state: SupervisorState) -> dict:
            agent_key = state["active_agent"]
            config = registry.get(agent_key)

            if not config:
                error_msg = (
                    f"'{agent_key}' 상담 서비스가 현재 이용 불가합니다. "
                    "다른 문의를 해주세요."
                )
                user_input = interrupt({"agent": "system", "message": error_msg})
                return {
                    "messages": [
                        AIMessage(content=error_msg),
                        HumanMessage(content=user_input),
                    ],
                }

            # 서브그래프 캐시에서 가져오기 (설정 변경 시 자동 재생성)
            sub_agent = agent_cache.get_or_create(config, model)

            # 서브그래프 실행: LLM → Tool → LLM → ... → 최종 응답
            result = await sub_agent.ainvoke({"messages": state["messages"]})

            # 최종 AI 응답 추출
            last_ai_content = ""
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    last_ai_content = msg.content
                    break

            # interrupt: 응답을 클라이언트에 전달하고 일시 중단
            user_input = interrupt(
                {"agent": agent_key, "message": last_ai_content}
            )

            # 서브그래프의 새 메시지 + 사용자 새 메시지를 반환
            return {
                "messages": result["messages"] + [HumanMessage(content=user_input)],
            }

        # ── 고정 그래프 구조 ──
        builder = StateGraph(SupervisorState)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("agent", agent_node)
        builder.add_edge(START, "supervisor")
        builder.add_edge("agent", "supervisor")

        self._graph = builder.compile(
            store=store,
            checkpointer=checkpointer,
        )

    async def ainvoke(self, question: str, session_id: str) -> dict:
        """새 대화를 시작합니다."""
        input_data = {
            "messages": [{"role": "user", "content": question}],
            "active_agent": "",
        }
        config = {"configurable": {"thread_id": session_id}}
        start_time = time.perf_counter()

        result = await self._graph.ainvoke(input_data, config)

        execution_time = time.perf_counter() - start_time

        state = await self._graph.aget_state(config)
        if state.next:
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "interrupt_info": await self._extract_interrupt_info(config),
            }

        return self._build_completed_response(result, execution_time)

    async def aresume(self, session_id: str, message: str) -> dict:
        """interrupt된 그래프를 사용자 메시지로 재개합니다."""
        config = {"configurable": {"thread_id": session_id}}

        state = await self._graph.aget_state(config)
        if not state.next:
            return {
                "status": "error",
                "message": "재개할 interrupt가 없습니다.",
            }

        start_time = time.perf_counter()

        result = await self._graph.ainvoke(Command(resume=message), config)

        execution_time = time.perf_counter() - start_time

        state = await self._graph.aget_state(config)
        if state.next:
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "interrupt_info": await self._extract_interrupt_info(config),
            }

        return self._build_completed_response(result, execution_time)

    async def _extract_interrupt_info(self, config: dict) -> dict:
        state = await self._graph.aget_state(config)
        interrupts = []
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    interrupts.append(
                        {
                            "value": intr.value,
                            "task_id": task.id,
                            "task_name": task.name,
                        }
                    )
        return {
            "next_nodes": list(state.next),
            "interrupts": interrupts,
        }

    @staticmethod
    def _build_completed_response(result: dict, execution_time: float) -> dict:
        answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break

        return {
            "status": "completed",
            "answer": answer,
            "active_agent": result.get("active_agent", ""),
            "execution_time": execution_time,
            "messages": result["messages"],
        }

    # ── Streaming ──

    async def astream_invoke(
        self, question: str, session_id: str
    ) -> AsyncIterator[dict]:
        """새 대화를 스트리밍으로 시작합니다."""
        input_data = {
            "messages": [{"role": "user", "content": question}],
            "active_agent": "",
        }
        config = {"configurable": {"thread_id": session_id}}
        async for event in self._astream_graph(input_data, config):
            yield event

    async def astream_resume(
        self, session_id: str, message: str
    ) -> AsyncIterator[dict]:
        """interrupt된 그래프를 스트리밍으로 재개합니다."""
        config = {"configurable": {"thread_id": session_id}}

        state = await self._graph.aget_state(config)
        if not state.next:
            yield {"type": "error", "message": "재개할 interrupt가 없습니다."}
            return

        async for event in self._astream_graph(Command(resume=message), config):
            yield event

    async def _astream_graph(
        self, input_data, config: dict
    ) -> AsyncIterator[dict]:
        """공통 스트리밍 로직.

        stream_mode=["messages", "updates"] + subgraphs=True 로
        create_agent 서브그래프의 LLM 토큰을 실시간 스트리밍합니다.
        """
        start_time = time.perf_counter()

        async for chunk in self._graph.astream(
            input_data,
            config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            # subgraphs=True → (namespace_tuple, mode, payload)
            if len(chunk) == 3:
                ns, mode, payload = chunk
            else:
                mode, payload = chunk
                ns = ()

            if mode == "messages":
                message, metadata = payload
                # 서브그래프(create_agent)의 AIMessageChunk만 스트리밍
                # ns가 비어있으면 부모 그래프(supervisor) → 필터링
                if (
                    isinstance(message, AIMessageChunk)
                    and message.content
                    and len(ns) > 0
                ):
                    yield {
                        "type": "token",
                        "content": message.content,
                    }

            elif mode == "updates":
                if "__interrupt__" in payload:
                    for intr_data in payload["__interrupt__"]:
                        value = (
                            intr_data.value
                            if hasattr(intr_data, "value")
                            else intr_data
                        )
                        yield {
                            "type": "interrupt",
                            "agent": (
                                value.get("agent", "")
                                if isinstance(value, dict)
                                else ""
                            ),
                            "message": (
                                value.get("message", "")
                                if isinstance(value, dict)
                                else str(value)
                            ),
                        }

        # 스트림 종료 후 최종 상태 확인
        execution_time = time.perf_counter() - start_time
        state = await self._graph.aget_state(config)

        if state.next:
            yield {
                "type": "end",
                "status": "interrupted",
                "execution_time": execution_time,
            }
        else:
            yield {
                "type": "end",
                "status": "completed",
                "execution_time": execution_time,
            }

    def get_graph(self) -> Any:
        return self._graph
