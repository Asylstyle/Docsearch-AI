"""Retrievers share one ready-document chunk snapshot; no QueryLog writes."""
from functools import lru_cache
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Chunk, Document
from app.schemas import SearchHit
from app.search.service import search_chunks
from sqlalchemy import select


def load_corpus(db):
    rows = db.execute(select(Chunk, Document.filename).join(Document)
                      .where(Document.status == "ready").order_by(Chunk.id)).all()
    if not rows:
        raise ValueError("No chunks from ready documents; ingest the corpus first")
    return [{"chunk_id": chunk.id, "document_id": chunk.document_id,
             "filename": filename, "page": chunk.page, "text": chunk.text}
            for chunk, filename in rows]


class TfidfRetriever:
    def __init__(self, corpus, variant):
        if variant == "tfidf_word":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        elif variant == "tfidf_char":
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        else:
            raise ValueError("Unknown TF-IDF variant")
        self.corpus = corpus
        self.matrix = self.vectorizer.fit_transform([row["text"] for row in corpus])

    def search(self, query, k):
        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix)[0]
        indexes = sorted(range(len(self.corpus)), key=lambda i: (-float(scores[i]), self.corpus[i]["chunk_id"]))[:k]
        return [SearchHit(**self.corpus[i], score=float(scores[i])) for i in indexes]


def e5_search(db, query, k):
    return search_chunks(db, query, k)


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache(maxsize=1)
def cross_encoder():
    return CrossEncoder(RERANKER_MODEL, device="cpu", trust_remote_code=False)


def hybrid_rrf(db, char_search, query, k=20):
    scores, hits = {}, {}
    for ranking in (e5_search(db, query, 20), char_search(query, 20)):
        for rank, hit in enumerate(ranking, 1):
            hits[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (60 + rank)
    ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:k]
    return [hits[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in ids]


def rerank(query, candidates):
    if not candidates:
        return []
    # Raw pairs: neither query: nor passage: prefixes belong to this model.
    scores = cross_encoder().predict([(query, hit.text) for hit in candidates], batch_size=16, show_progress_bar=False)
    order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i].chunk_id))
    return [candidates[i].model_copy(update={"score": float(scores[i])}) for i in order]


def e5_rerank(db, query, k=5):
    candidates = e5_search(db, query, 20)
    return rerank(query, candidates)[:k], candidates


def hybrid_rerank(db, char_search, query, k=5):
    candidates = hybrid_rrf(db, char_search, query, 20)
    return rerank(query, candidates)[:k], candidates
