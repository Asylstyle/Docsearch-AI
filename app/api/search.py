from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import QueryLog
from app.schemas import SearchHit
from app.search.service import search_chunks

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SearchHit])
def search(q: str, k: int = Query(default=5, ge=1), db: Session = Depends(get_db)):
    if not q.strip() or len(q) > 2000 or k > 100:
        raise HTTPException(422, "Require nonempty q up to 2000 characters and k between 1 and 100")
    try:
        hits = search_chunks(db, q, k)
        db.add(QueryLog(query_type="search", question=q, answer=None,
                       sources=[{"document_id": hit.document_id, "filename": hit.filename, "page": hit.page}
                                for hit in hits]))
        db.commit()
        return hits
    except Exception as exc:
        db.rollback()
        raise HTTPException(503, "Search unavailable") from exc
