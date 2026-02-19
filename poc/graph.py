import time
from typing import Optional, Any

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from poc.state import SupervisorState
from poc.registry import AgentRegistry
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


class SupervisorGraphOrchestrator:
    """Dynamic Supervisor 기반 멀티 에이전트 오케스트레이터

    그래프 구조는 고정 (supervisor ↔ agent),
    에이전트 행동은 AgentRegistry에서 동적으로 결정됩니다.

    - 에이전트 추가/삭제 시 그래프 재컴파일 불필요
    - Supervisor 프롬프트와 라우팅 스키마가 매 호출마다 레지스트리에서 동적 생성
    - 스키마/프롬프트는 레지스트리 version 기반 캐싱으로 불필요한 재생성 방지

    그래프 플로우:
        START → supervisor → agent → (interrupt/resume) → supervisor → ... → END
    """

    def __init__(self, registry: AgentRegistry):
        self._graph: Optional[Any] = None
        self._registry = registry
        self._llm_manager = get_llm_manager()

    async def initialize(self, store, checkpointer) -> None:
        """그래프를 빌드하고 컴파일합니다."""
        model = self._llm_manager.get_model(ModelName.GPT_4O_MINI)
        registry = self._registry

        # 캐시: 레지스트리 버전이 바뀔 때만 스키마/프롬프트 재생성
        _cache: dict = {"version": -1, "llm": None, "prompt": ""}

        def _refresh_cache() -> None:
            """레지스트리 변경 시 supervisor LLM과 프롬프트를 갱신합니다."""
            if registry.version == _cache["version"]:
                return
            schema = registry.build_route_schema()
            _cache["llm"] = model.with_structured_output(schema)
            _cache["prompt"] = SUPERVISOR_PROMPT_TEMPLATE.format(
                agent_descriptions=registry.get_descriptions()
            )
            _cache["version"] = registry.version

        # ── Supervisor 노드 ──
        async def supervisor_node(state: SupervisorState) -> Command:
            """사용자 메시지를 분석하여 적절한 에이전트로 라우팅합니다."""
            _refresh_cache()

            messages = [SystemMessage(content=_cache["prompt"])] + state["messages"]
            decision = await _cache["llm"].ainvoke(messages)

            next_agent = decision.next.value  # Enum → str

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

        # ── 범용 에이전트 노드 ──
        async def agent_node(state: SupervisorState) -> dict:
            """active_agent에 해당하는 프롬프트를 동적 로드하여 응답합니다."""
            agent_key = state["active_agent"]
            config = registry.get(agent_key)

            if not config:
                # 세션 도중 에이전트가 삭제된 경우
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

            messages = [SystemMessage(content=config.prompt)] + state["messages"]
            response: AIMessage = await model.ainvoke(messages)

            user_input = interrupt(
                {
                    "agent": agent_key,
                    "message": response.content,
                }
            )

            return {
                "messages": [response, HumanMessage(content=user_input)],
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
        """중단된 그래프에서 interrupt 정보를 추출합니다."""
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
        """완료된 그래프 결과를 응답 형태로 구성합니다."""
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

    def get_graph(self) -> Any:
        """컴파일된 그래프를 반환합니다."""
        return self._graph
