from app.schemas import SearchHit


def embed_texts(texts: list[str]) -> list[list[float]]:
    raise NotImplementedError


def embed_query(query: str) -> list[float]:
    raise NotImplementedError


def search_chunks(db, query, k=5) -> list[SearchHit]:
    raise NotImplementedError
