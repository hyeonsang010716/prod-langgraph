from typing import Optional, Any, Dict, Tuple, List
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.callbacks import get_openai_callback
import time
from functools import lru_cache

from app.core.graph.example.graph_state import GraphState, SubGraphState
from app.core.graph.example.prompt_manager import PromptManager
from app.core.graph.example.chain_builder import ChainManager


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

    def _analyze_question(self, state: SubGraphState) -> Dict[str, Any]:
        """질문을 분석합니다."""
        question_text = state['question'].content
        analysis = f"'{question_text}'에 대한 분석 완료"
        print(f"  [SubGraph - Analyze] {analysis}")
        return {"analysis": analysis}

    def _generate_response(self, state: SubGraphState) -> Dict[str, Any]:
        """분석 결과를 기반으로 응답을 생성합니다."""
        answer = AIMessage(content=f"[SubGraph 응답] {state['analysis']} → 답변 생성 완료")
        print(f"  [SubGraph - Response] {answer.content}")
        return {"answer": answer}

    def _build_sub_graph(self) -> Any:
        """서브 그래프를 구성합니다."""
        sub_builder = StateGraph(SubGraphState)

        sub_builder.add_node("AnalyzeQuestion", self._analyze_question)
        sub_builder.add_node("GenerateResponse", self._generate_response)

        sub_builder.add_edge(START, "AnalyzeQuestion")
        sub_builder.add_edge("AnalyzeQuestion", "GenerateResponse")
        sub_builder.add_edge("GenerateResponse", END)

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


    async def ainvoke(
        self, 
        question: str, 
        session_id: str
    ) -> Tuple[str, float, int, float, List]:
        """그래프를 실행합니다.
        
        Returns:
            tuple: (answer, execution_time, total_tokens, total_cost, history)
        """
        input_data = {"question": HumanMessage(content=question)}
        config = {"configurable": {"thread_id": session_id}}
            
        start_time = time.perf_counter()
        
        with get_openai_callback() as cb:
            response = await self._graph.ainvoke(input_data, config)
        
        end_time = time.perf_counter()
    
        answer_content = response['answer'].content
        
        return (
            answer_content,
            end_time - start_time, 
            cb.total_tokens, 
            cb.total_cost,
            response['messages']
        )
    
    
@lru_cache(maxsize=1)
def get_example_graph() -> GraphOrchestrator:
    """GraphOrchestrator 싱글톤 인스턴스를 반환합니다."""
    prompt_manager = PromptManager()
    chain_manager = ChainManager(prompt_manager)
    return GraphOrchestrator(
        prompt_manager=prompt_manager,
        chain_manager=chain_manager
    )