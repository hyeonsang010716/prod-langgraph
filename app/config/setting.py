from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 서버
    HOST: str = Field("0.0.0.0", description="서버 호스트")
    PORT: int = Field(8000, description="서버 포트 번호") 

    # LLM KEY
    OPENAI_API_KEY: str = Field("sk-", description="OpenAI API KEY")
    
    # POSTGRES 정보
    POSTGRES_HOST: str = Field("hyeonsang-postgres", description="POSTGRES HOST")
    POSTGRES_PORT: int = Field(5432, description="POSTGRES PORT")
    POSTGRES_USER: str = Field("cho", description="POSTGRES USER")
    POSTGRES_PASSWORD: str = Field("hyeonsang", description="POSTGRES PASSWORD")
    POSTGRES_NAME: str = Field("chohyeonsang", description="POSTGRES NAME")
    
    # Langsmith
    LANGCHAIN_TRACING_V2: Optional[bool] = None
    LANGCHAIN_ENDPOINT: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    
    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_NAME}"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()