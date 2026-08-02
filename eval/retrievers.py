"""Retrievers share one ready-document chunk snapshot; no QueryLog writes."""
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
