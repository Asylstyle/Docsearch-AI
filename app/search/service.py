from functools import lru_cache
from threading import RLock

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


def _e5_search(db, query, k=5) -> list[SearchHit]:
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


_index_lock = RLock()
_char_cache = None


def invalidate_search_index():
    global _char_cache
    with _index_lock:
        _char_cache = None


def _char_search(db, query, k):
    global _char_cache
    with _index_lock:
        if _char_cache is None:
            rows = db.execute(select(Chunk, Document.filename).join(Document)
                .where(Document.status == 'ready').order_by(Chunk.id)).all()
            if not rows:
                return []
            hits = [SearchHit(chunk_id=c.id, document_id=c.document_id, filename=f,
                              page=c.page, text=c.text, score=0) for c, f in rows]
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), lowercase=True)
            matrix = vectorizer.fit_transform([hit.text for hit in hits])
            _char_cache = hits, vectorizer, matrix
        hits, vectorizer, matrix = _char_cache
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), hits[i].chunk_id))[:k]
    return [hits[i].model_copy(update={'score':float(scores[i])}) for i in order]


@lru_cache(maxsize=1)
def _reranker():
    return CrossEncoder(settings.RERANK_MODEL, device='cpu', trust_remote_code=False)


def search_chunks(db, query, k=5) -> list[SearchHit]:
    if not query.strip() or k < 1:
        raise ValueError('Query must not be empty and k must be positive')
    mode = settings.SEARCH_MODE
    if mode == 'e5':
        return _e5_search(db, query, k)
    count = settings.RERANK_CANDIDATES
    candidates = _e5_search(db, query, count)
    if mode in ('hybrid_rrf', 'hybrid_rerank'):
        ranks = (candidates, _char_search(db, query, count))
        scores, hits = {}, {}
        for ranking in ranks:
            for rank, hit in enumerate(ranking, 1):
                hits[hit.chunk_id] = hit
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0) + 1/(60+rank)
        ids = sorted(scores, key=lambda i: (-scores[i], i))[:count]
        candidates = [hits[i].model_copy(update={'score':scores[i]}) for i in ids]
    if mode in ('e5_rerank', 'hybrid_rerank') and candidates:
        scores = _reranker().predict([(query, hit.text) for hit in candidates],
                                     batch_size=16, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i].chunk_id))
        candidates = [candidates[i].model_copy(update={'score':float(scores[i])}) for i in order]
    return candidates[:k]
