from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.config import settings
from app.models import Chunk, Document
from app.schemas import SearchHit


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(["passage: " + text for text in texts], normalize_embeddings=True)
    if vectors.shape[1] != settings.EMBEDDING_DIM or vectors.shape[1] != 384:
        raise ValueError("Embedding dimension must be 384")
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    vector = _model().encode("query: " + query, normalize_embeddings=True)
    if len(vector) != settings.EMBEDDING_DIM or len(vector) != 384:
        raise ValueError("Embedding dimension must be 384")
    return vector.tolist()


def search_chunks(db, query, k=5) -> list[SearchHit]:
    if not query.strip() or k < 1:
        raise ValueError("Query must not be empty and k must be positive")
    distance = Chunk.embedding.cosine_distance(embed_query(query))
    rows = db.execute(select(Chunk, Document.filename, (1 - distance).label("score"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == "ready")
        .order_by(distance, Chunk.id).limit(k)).all()
    return [SearchHit(chunk_id=chunk.id, document_id=chunk.document_id, filename=filename,
                      page=chunk.page, text=chunk.text, score=float(score))
            for chunk, filename, score in rows]
