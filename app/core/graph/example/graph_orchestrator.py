from typing import Optional, Any, Dict, Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.callbacks import get_openai_callback
import time
from functools import lru_cache

from app.core.graph.example.graph_state import GraphState
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

        
        
    def _add_history_message(self, state: GraphState) -> Dict[str, Any]:
        """히스토리 메시지를 추가합니다."""
        
        return {"messages": [state['question'] , AIMessage(content="테스트 답변")] , "answer" : AIMessage(content="테스트 답변")}
    
    
    async def _build_graph(self, store, checkpointer) -> None:
        """LangGraph를 구성합니다."""
        graph_builder = StateGraph(GraphState)
        
        # 노드 추가
        graph_builder.add_node("AddHistoryMessage", self._add_history_message)
        
        # 엣지 추가
        graph_builder.add_edge(START, "AddHistoryMessage")
        graph_builder.add_edge("AddHistoryMessage", END)
        
        # 그래프 컴파일
        self._graph = graph_builder.compile(
            store=store,
            checkpointer=checkpointer                                    
        )
    
    
    def get_graph(self) -> Any:
        """컴파일된 그래프를 반환합니다."""
        return self._graph
    
    
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