from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import AnswerResult, AskRequest

router = APIRouter(tags=["questions"])


@router.post("/ask", response_model=AnswerResult, responses={501: {"description": "Scaffold only"}})
def ask(request: AskRequest, db: Session = Depends(get_db)):
    raise NotImplementedError


@router.get("/history", responses={501: {"description": "Scaffold only"}})
def history(limit: int = Query(default=20, ge=1), db: Session = Depends(get_db)):
    raise NotImplementedError
