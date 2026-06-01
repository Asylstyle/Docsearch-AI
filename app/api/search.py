from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import SearchHit

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SearchHit], responses={501: {"description": "Scaffold only"}})
def search(q: str, k: int = Query(default=5, ge=1), db: Session = Depends(get_db)):
    raise NotImplementedError
