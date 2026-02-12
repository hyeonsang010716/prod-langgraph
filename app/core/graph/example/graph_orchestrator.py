from typing import Optional, Any, Dict, List, Literal
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, Send
from langchain_community.callbacks import get_openai_callback
import time
from functools import lru_cache

from app.core.graph.example.graph_state import GraphState, SubGraphState
from app.core.graph.example.prompt_manager import PromptManager
from app.core.graph.example.chain_builder import ChainManager


# Send fan-out에 사용할 분석 관점들
ANALYSIS_PERSPECTIVES = ["감정분석", "주제분류", "키워드추출"]



class GraphOrchestrator:
    """LangGraph 오케스트레이터 클래스"""

    def __init__(
        self,
        prompt_manager: PromptManager,
        chain_manager: ChainManager
    ):
        self._prompt_manager = prompt_manager
        self._chain_manager = chain_manager
        self._graph: Optional[Any] = None


    async def initialize(self, store, checkpointer) -> None:
        """그래프와 관련 컴포넌트를 초기화합니다."""

        # 모든 체인 빌드
        self._chain_manager.build_all_chains()

        # 그래프 생성
        await self._build_graph(store, checkpointer)



    # ===== 서브 그래프 노드 =====

    def _fan_out_analysis(self, state: SubGraphState) -> List[Send]:
        """Send를 사용하여 여러 관점으로 병렬 분석을 fan-out합니다.

        각 관점(감정분석, 주제분류, 키워드추출)마다 AnalyzeWorker 노드로 Send를 보냅니다.
        Send는 같은 노드를 서로 다른 인자로 병렬 실행합니다.
        """
        question_text = state['question'].content
        print(f"  [SubGraph - FanOut] '{question_text}' → {len(ANALYSIS_PERSPECTIVES)}개 관점으로 병렬 분석 시작")

        return [
            Send("AnalyzeWorker", {
                "question": state['question'],
                "perspective": perspective,
            })
            for perspective in ANALYSIS_PERSPECTIVES
        ]

    def _analyze_worker(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """개별 관점으로 질문을 분석합니다. (Send로 병렬 실행됨)

        Send의 arg로 전달된 dict가 state로 들어옵니다.
        결과는 analyses 리스트에 operator.add 리듀서로 합쳐집니다.
        """
        question_text = state['question'].content
        perspective = state['perspective']

        result = f"[{perspective}] '{question_text}' 분석 결과"
        print(f"  [SubGraph - Worker] {result}")

        # analyses는 Annotated[List[str], operator.add]이므로 리스트로 반환 → 자동 병합
        return {"analyses": [result]}

    def _merge_analyses(self, state: SubGraphState) -> Dict[str, Any]:
        """병렬 분석 결과를 하나의 analysis 문자열로 병합합니다."""
        analyses = state['analyses']
        merged = " | ".join(analyses)
        print(f"  [SubGraph - Merge] {len(analyses)}개 분석 병합 완료")
        return {"analysis": merged}

    def _human_review(self, state: SubGraphState) -> Dict[str, Any]:
        """사용자 검토를 요청합니다. (interrupt 발생 지점)"""
        analysis = state['analysis']
        print(f"  [SubGraph - HumanReview] interrupt 발생, 사용자 검토 대기중...")

        user_decision = interrupt({
            "analysis": analysis,
            "message": "분석 결과를 검토해주세요. 'approve'로 승인하거나, 수정된 분석 내용을 입력하세요.",
        })

        if user_decision == "approve":
            print(f"  [SubGraph - HumanReview] 승인됨, 기존 분석 유지")
            return {"analysis": analysis}
        else:
            print(f"  [SubGraph - HumanReview] 수정됨: {user_decision}")
            return {"analysis": user_decision}

    def _route_response(self, state: SubGraphState) -> Command[Literal["SimpleResponse", "DetailedResponse"]]:
        """Command(goto=...)를 사용하여 분석 결과에 따라 동적 라우팅합니다.

        반환 타입의 Literal[...]이 그래프 빌더에게 도달 가능한 노드를 알려줍니다.
        분석 결과가 길면(100자 초과) → DetailedResponse (상세 응답)
        짧으면 → SimpleResponse (간단 응답)
        """
        analysis = state['analysis']

        if len(analysis) > 100:
            print(f"  [SubGraph - Route] 분석 길이 {len(analysis)}자 → DetailedResponse로 라우팅")
            return Command(goto="DetailedResponse")
        else:
            print(f"  [SubGraph - Route] 분석 길이 {len(analysis)}자 → SimpleResponse로 라우팅")
            return Command(goto="SimpleResponse")

    def _simple_response(self, state: SubGraphState) -> Dict[str, Any]:
        """간단한 응답을 생성합니다."""
        answer = AIMessage(content=f"[간단 응답] {state['analysis']}")
        print(f"  [SubGraph - SimpleResponse] {answer.content}")
        return {"answer": answer}

    def _detailed_response(self, state: SubGraphState) -> Dict[str, Any]:
        """상세한 응답을 생성합니다."""
        analyses = state.get('analyses', [])
        detail_parts = [f"  - {a}" for a in analyses]
        detail = "\n".join(detail_parts)
        answer = AIMessage(
            content=f"[상세 응답]\n종합 분석: {state['analysis']}\n\n세부 분석:\n{detail}"
        )
        print(f"  [SubGraph - DetailedResponse] {answer.content}")
        return {"answer": answer}

    def _build_sub_graph(self) -> Any:
        """서브 그래프를 구성합니다.

        플로우:
          START ─(Send fan-out)─→ AnalyzeWorker (병렬 3개)
                                      ↓ (모두 완료 후, operator.add로 analyses 병합)
                                 MergeAnalyses
                                      ↓
                                 HumanReview (interrupt)
                                      ↓
                                 RouteResponse ─(Command goto)─→ SimpleResponse → END
                                                              → DetailedResponse → END
        """
        sub_builder = StateGraph(SubGraphState)

        # 노드 추가 (FanOutAnalysis는 노드가 아님 - 라우팅 함수로만 사용)
        sub_builder.add_node("AnalyzeWorker", self._analyze_worker)
        sub_builder.add_node("MergeAnalyses", self._merge_analyses)
        sub_builder.add_node("HumanReview", self._human_review)
        sub_builder.add_node("RouteResponse", self._route_response)
        sub_builder.add_node("SimpleResponse", self._simple_response)
        sub_builder.add_node("DetailedResponse", self._detailed_response)

        # Send fan-out: START에서 _fan_out_analysis 라우팅 함수가 List[Send]를 반환
        # → AnalyzeWorker 노드가 3개 관점으로 병렬 실행됨
        sub_builder.add_conditional_edges(START, self._fan_out_analysis)

        # Worker 완료 후 병합
        sub_builder.add_edge("AnalyzeWorker", "MergeAnalyses")

        # 병합 → 사용자 검토 → 라우팅
        sub_builder.add_edge("MergeAnalyses", "HumanReview")
        sub_builder.add_edge("HumanReview", "RouteResponse")
        # RouteResponse는 Command(goto=...)를 반환하므로 명시적 엣지 불필요
        # Command[Literal["SimpleResponse", "DetailedResponse"]] 타입 어노테이션이 그래프에 알려줌

        # 응답 → END
        sub_builder.add_edge("SimpleResponse", END)
        sub_builder.add_edge("DetailedResponse", END)

        return sub_builder.compile()

    # ===== 메인 그래프 노드 =====

    def _add_history_message(self, state: GraphState) -> Dict[str, Any]:
        """히스토리 메시지를 추가합니다."""
        return {"messages": [state['question'], state['answer']]}


    async def _build_graph(self, store, checkpointer) -> None:
        """LangGraph를 구성합니다."""
        graph_builder = StateGraph(GraphState)

        # 서브 그래프 빌드 & 노드 추가
        sub_graph = self._build_sub_graph()
        graph_builder.add_node("ProcessSubGraph", sub_graph)
        graph_builder.add_node("AddHistoryMessage", self._add_history_message)

        # 엣지 추가
        graph_builder.add_edge(START, "ProcessSubGraph")
        graph_builder.add_edge("ProcessSubGraph", "AddHistoryMessage")
        graph_builder.add_edge("AddHistoryMessage", END)

        # 그래프 컴파일
        self._graph = graph_builder.compile(
            store=store,
            checkpointer=checkpointer
        )
    
    
    def get_graph(self) -> Any:
        """컴파일된 그래프를 반환합니다."""
        return self._graph


    # ===== 메시지 삭제 =====

    async def adelete_messages(self, session_id: str, message_ids: List[str]) -> dict:
        """특정 메시지를 삭제합니다."""
        config = {"configurable": {"thread_id": session_id}}

        state = await self._graph.aget_state(config)
        messages = state.values.get('messages', [])

        if not messages:
            return {"deleted": 0, "remaining": 0}

        existing_ids = {msg.id for msg in messages}
        valid_ids = [mid for mid in message_ids if mid in existing_ids]

        if not valid_ids:
            return {"deleted": 0, "remaining": len(messages)}

        remove_messages = [RemoveMessage(id=mid) for mid in valid_ids]
        await self._graph.aupdate_state(config, {"messages": remove_messages})

        return {"deleted": len(valid_ids), "remaining": len(messages) - len(valid_ids)}

    async def aclear_history(self, session_id: str) -> dict:
        """세션의 전체 대화 히스토리를 삭제합니다."""
        config = {"configurable": {"thread_id": session_id}}

        state = await self._graph.aget_state(config)
        messages = state.values.get('messages', [])

        if not messages:
            return {"deleted": 0}

        remove_messages = [RemoveMessage(id=msg.id) for msg in messages]
        await self._graph.aupdate_state(config, {"messages": remove_messages})

        return {"deleted": len(messages)}


    async def _extract_interrupt_info(self, config: dict) -> dict:
        """현재 상태에서 interrupt 정보를 추출합니다."""
        state = await self._graph.aget_state(config)
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                return task.interrupts[0].value
        return {}

    async def ainvoke(self, question: str, session_id: str) -> dict:
        """그래프를 실행합니다. interrupt 발생 시 중단 정보를 반환합니다."""
        input_data = {"question": HumanMessage(content=question)}
        config = {"configurable": {"thread_id": session_id}}

        start_time = time.perf_counter()

        with get_openai_callback() as cb:
            response = await self._graph.ainvoke(input_data, config)

        execution_time = time.perf_counter() - start_time

        # interrupt 여부 확인: aget_state의 next가 있으면 아직 실행할 노드가 남아있음
        state = await self._graph.aget_state(config)
        if state.next:
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "interrupt_info": await self._extract_interrupt_info(config),
            }

        return {
            "status": "completed",
            "answer": response['answer'].content,
            "execution_time": execution_time,
            "total_tokens": cb.total_tokens,
            "total_cost": cb.total_cost,
            "messages": response['messages'],
        }

    async def aresume(self, session_id: str, action: str) -> dict:
        """interrupt된 그래프를 재개합니다."""
        config = {"configurable": {"thread_id": session_id}}

        start_time = time.perf_counter()

        with get_openai_callback() as cb:
            response = await self._graph.ainvoke(
                Command(resume=action), config
            )

        execution_time = time.perf_counter() - start_time

        state = await self._graph.aget_state(config)
        if state.next:
            return {
                "status": "interrupted",
                "execution_time": execution_time,
                "interrupt_info": await self._extract_interrupt_info(config),
            }

        return {
            "status": "completed",
            "answer": response['answer'].content,
            "execution_time": execution_time,
            "total_tokens": cb.total_tokens,
            "total_cost": cb.total_cost,
            "messages": response['messages'],
        }


@lru_cache(maxsize=1)
def get_example_graph() -> GraphOrchestrator:
    """GraphOrchestrator 싱글톤 인스턴스를 반환합니다."""
    prompt_manager = PromptManager()
    chain_manager = ChainManager(prompt_manager)
    return GraphOrchestrator(
        prompt_manager=prompt_manager,
        chain_manager=chain_manager
    )