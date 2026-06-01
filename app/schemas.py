from pydantic import BaseModel, Field


class ChunkData(BaseModel):
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    page: int = Field(ge=1)
    text: str
    score: float


class SourceRef(BaseModel):
    document_id: int
    filename: str
    page: int = Field(ge=1)


class AnswerResult(BaseModel):
    answer: str
    sources: list[SourceRef]


class AskRequest(BaseModel):
    question: str
    k: int = Field(default=5, ge=1)
