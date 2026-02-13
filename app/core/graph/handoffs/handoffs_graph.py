import time
from typing import Optional, Any, Literal
from functools import lru_cache

from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_community.callbacks import get_openai_callback
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

from app.core.graph.handoffs.handoffs_state import HandoffsState
from app.core.llm_manager import ModelName, get_llm_manager


# ===== Transfer Tools =====
# 각 tool은 Command(graph=Command.PARENT)를 반환하여
# 부모 그래프 레벨에서 다른 에이전트 노드로 라우팅합니다.
# ToolRuntime은 create_agent 내부의 ToolNode에 의해 자동 주입됩니다.

@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """고객이 상품 문의, 가격, 구매에 대해 물어볼 때 Sales 에이전트로 전환합니다."""
    # AIMessage(tool_call) + ToolMessage 쌍을 포함해야 대화 히스토리가 유효합니다.
    last_ai = next(
        msg for msg in reversed(runtime.state["messages"])
        if isinstance(msg, AIMessage)
    )
    transfer_msg = ToolMessage(
        content="Sales 에이전트로 전환되었습니다.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai, transfer_msg],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_support(runtime: ToolRuntime) -> Command:
    """고객이 기술 지원, 장애, 사용법에 대해 물어볼 때 Support 에이전트로 전환합니다."""
    last_ai = next(
        msg for msg in reversed(runtime.state["messages"])
        if isinstance(msg, AIMessage)
    )
    transfer_msg = ToolMessage(
        content="Support 에이전트로 전환되었습니다.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai, transfer_msg],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_support_with_confirm(runtime: ToolRuntime) -> Command:
    """고객 확인 후 Support 에이전트로 전환합니다. 민감한 기술 이슈(데이터 복구, 계정 삭제 등)에 사용합니다."""
    last_ai = next(
        msg for msg in reversed(runtime.state["messages"])
        if isinstance(msg, AIMessage)
    )

    # interrupt: 그래프 실행을 중단하고 사용자 확인을 기다림
    # 클라이언트가 Command(resume=값)으로 재개하면 그 값이 반환됨
    user_decision = interrupt(
        "기술 지원팀으로 전환하시겠습니까? 전환 시 대화 내역이 공유됩니다. (승인/거절)"
    )

    if user_decision == "거절":
        # 거절 시 현재 에이전트에 머무름 (핸드오프 취소)
        transfer_msg = ToolMessage(
            content="전환이 취소되었습니다. 현재 에이전트가 계속 도와드리겠습니다.",
            tool_call_id=runtime.tool_call_id,
        )
        return Command(
            update={"messages": [last_ai, transfer_msg]},
            graph=Command.PARENT,
        )

    # 승인 시 Support 에이전트로 핸드오프
    transfer_msg = ToolMessage(
        content="고객 확인 완료. Support 에이전트로 전환되었습니다.",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai, transfer_msg],
        },
        graph=Command.PARENT,
    )


# ===== System Prompts =====

SALES_PROMPT = """당신은 Sales 에이전트입니다.
고객의 상품 문의, 가격 안내, 구매 상담을 담당합니다.

역할:
- 상품 추천 및 비교
- 가격과 할인 정보 안내
- 구매 절차 안내

전환 규칙:
- 일반 기술 문의 → transfer_to_support 사용
- 민감한 기술 이슈(데이터 복구, 계정 삭제, 보안 문제 등) → transfer_to_support_with_confirm 사용 (고객 확인 필요)
항상 한국어로 답변하세요."""

SUPPORT_PROMPT = """당신은 Support 에이전트입니다.
고객의 기술 지원, 장애 해결, 사용법 안내를 담당합니다.

역할:
- 기술적 문제 진단 및 해결
- 서비스 사용법 안내
- 장애 상황 대응

고객이 상품 구매나 가격 문의를 하면, transfer_to_sales 도구를 사용하여 Sales 에이전트로 전환하세요.
항상 한국어로 답변하세요."""


class HandoffsGraphOrchestrator:
    """멀티 에이전트 Handoffs 오케스트레이터

    Sales / Support 두 에이전트가 서로 핸드오프하며 대화를 처리합니다.
    각 에이전트는 create_agent로 생성된 서브그래프이며,
    transfer 도구의 Command(graph=PARENT)로 부모 그래프에서 라우팅됩니다.

    그래프 플로우:
        START → (route_initial) → sales_agent ⇄ support_agent → END
    """

    def __init__(self):
        self._graph: Optional[Any] = None
        self._llm_manager = get_llm_manager()

    async def initialize(self, store, checkpointer) -> None:
        """에이전트 그래프를 빌드하고 컴파일합니다."""
        model = self._llm_manager.get_model(ModelName.GPT_4O_MINI)

        # create_agent: langchain v1의 표준 에이전트 팩토리
        # - system_prompt: 에이전트 역할 정의
        # - tools: 사용 가능한 도구 목록
        # - 반환: CompiledStateGraph (서브그래프로 사용 가능)
        sales_agent = create_agent(
            model=model,
            tools=[transfer_to_support, transfer_to_support_with_confirm],
            system_prompt=SALES_PROMPT,
            name="sales_agent",
        )
        support_agent = create_agent(
            model=model,
            tools=[transfer_to_sales],
            system_prompt=SUPPORT_PROMPT,
            name="support_agent",
        )

        # 에이전트 노드 래퍼: 서브그래프를 invoke하고 결과를 반환
        def call_sales_agent(state: HandoffsState) -> Command:
            """Sales 에이전트를 호출합니다."""
            return sales_agent.invoke(state)

        def call_support_agent(state: HandoffsState) -> Command:
            """Support 에이전트를 호출합니다."""
            return support_agent.invoke(state)

        # 부모 그래프 빌드
        builder = StateGraph(HandoffsState)
        builder.add_node("sales_agent", call_sales_agent)
        builder.add_node("support_agent", call_support_agent)

        # 초기 라우팅: active_agent에 따라 진입점 결정
        builder.add_conditional_edges(
            START,
            self._route_initial,
            ["sales_agent", "support_agent"],
        )

        # 각 에이전트 종료 후 라우팅: 핸드오프 또는 대화 종료
        builder.add_conditional_edges(
            "sales_agent",
            self._route_after_agent,
            ["sales_agent", "support_agent", END],
        )
        builder.add_conditional_edges(
            "support_agent",
            self._route_after_agent,
            ["sales_agent", "support_agent", END],
        )

        self._graph = builder.compile(
            store=store,
            checkpointer=checkpointer,
        )

    @staticmethod
    def _route_initial(
        state: HandoffsState,
    ) -> Literal["sales_agent", "support_agent"]:
        """초기 진입 에이전트를 결정합니다."""
        return state.get("active_agent") or "sales_agent"

    @staticmethod
    def _route_after_agent(
        state: HandoffsState,
    ) -> Literal["sales_agent", "support_agent", "__end__"]:
        """에이전트 실행 후 다음 행선지를 결정합니다.

        - 마지막 메시지가 tool_call 없는 AIMessage → 대화 종료 (END)
        - 그 외 → active_agent로 라우팅 (핸드오프 발생)
        """
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                return "__end__"
        return state.get("active_agent", "sales_agent")

    async def ainvoke(self, question: str, session_id: str) -> dict:
        """그래프를 실행합니다. interrupt 발생 시 중단 정보를 반환합니다."""
        input_data = {
            "messages": [{"role": "user", "content": question}],
            "active_agent": "sales_agent",
        }
        config = {"configurable": {"thread_id": session_id}}
        start_time = time.perf_counter()

        with get_openai_callback() as cb:
            result = await self._graph.ainvoke(input_data, config)

        execution_time = time.perf_counter() - start_time

        # interrupt 여부 확인: state.next가 있으면 아직 실행할 노드가 남아있음
        state = await self._graph.aget_state(config)
        if state.next:
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "interrupt_info": await self._extract_interrupt_info(config),
            }

        return self._build_completed_response(result, execution_time, cb)

    async def aresume(self, session_id: str, action: str) -> dict:
        """interrupt된 그래프를 재개합니다."""
        config = {"configurable": {"thread_id": session_id}}

        # 재개 가능 여부 사전 검증
        state = await self._graph.aget_state(config)
        if not state.next:
            return {
                "status": "error",
                "message": "재개할 interrupt가 없습니다. 이미 처리되었거나 존재하지 않는 세션입니다.",
            }

        start_time = time.perf_counter()

        with get_openai_callback() as cb:
            result = await self._graph.ainvoke(
                Command(resume=action), config
            )

        execution_time = time.perf_counter() - start_time

        # 재개 후에도 또 interrupt가 발생할 수 있음
        state = await self._graph.aget_state(config)
        if state.next:
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "interrupt_info": await self._extract_interrupt_info(config),
            }

        return self._build_completed_response(result, execution_time, cb)

    async def _extract_interrupt_info(self, config: dict) -> dict:
        """중단된 그래프에서 interrupt 정보를 추출합니다."""
        state = await self._graph.aget_state(config)
        interrupts = []
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for intr in task.interrupts:
                    interrupts.append({
                        "value": intr.value,
                        "interrupt_id": intr.id,
                        "task_id": task.id,
                        "task_name": task.name,
                    })
        return {
            "next_nodes": list(state.next),
            "interrupts": interrupts,
        }

    @staticmethod
    def _build_completed_response(result: dict, execution_time: float, cb) -> dict:
        """완료된 그래프 결과를 응답 형태로 구성합니다."""
        answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                answer = msg.content
                break

        return {
            "status": "completed",
            "answer": answer,
            "active_agent": result.get("active_agent", "sales_agent"),
            "execution_time": execution_time,
            "total_tokens": cb.total_tokens,
            "total_cost": cb.total_cost,
            "messages": result["messages"],
        }

    def get_graph(self) -> Any:
        """컴파일된 그래프를 반환합니다."""
        return self._graph


@lru_cache(maxsize=1)
def get_handoffs_graph() -> HandoffsGraphOrchestrator:
    """HandoffsGraphOrchestrator 싱글톤 인스턴스를 반환합니다."""
    return HandoffsGraphOrchestrator()
