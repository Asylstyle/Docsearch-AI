from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://docsearch:docsearch@db:5432/docsearch"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIM: int = 384
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-5.6-terra"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    SEARCH_MODE: Literal['e5', 'hybrid_rrf', 'e5_rerank', 'hybrid_rerank'] = 'e5'
    RERANK_MODEL: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    RERANK_CANDIDATES: int = Field(default=20, ge=5, le=100)


settings = Settings()
