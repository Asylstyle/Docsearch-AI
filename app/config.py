from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg://docsearch:docsearch@db:5432/docsearch"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIM: int = 384
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100


settings = Settings()
