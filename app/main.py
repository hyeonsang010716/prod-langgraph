from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.config.setting import settings
from app.core.llm_manager import get_llm_manager
# from app.core.graph.example.graph_orchestrator import get_example_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    
    # LLM 초기화
    get_llm_manager()
    
    # example_graph = get_example_graph()
    # await example_graph.initialize()
    
    yield
    
    # example_graph = get_example_graph()
    # await example_graph.cleanup()


def create_app() -> FastAPI:
    """FastAPI 애플리케이션 팩토리"""
    
    app = FastAPI(
        title="FastAPI Server",
        description="FastAPI 서버 공통 세팅",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Frontend URL 권한 부여
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_config=None,
    )