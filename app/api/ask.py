from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db import get_db
from app.models import QueryLog
from app.rag.service import answer_question
from app.schemas import AnswerResult, AskRequest, HistoryItem

router = APIRouter(tags=["questions"])


@router.post("/ask", response_model=AnswerResult, responses={503: {"description": "Retrieval or OpenAI unavailable"}})
def ask(request: AskRequest, db: Session = Depends(get_db)):
    if not request.question.strip() or len(request.question) > 2000 or request.k > 20:
        raise HTTPException(422, "Require nonempty question up to 2000 characters and k between 1 and 20")
    try:
        result = answer_question(db, request.question, request.k)
        db.add(QueryLog(query_type="ask", question=request.question, answer=result.answer,
                       sources=[source.model_dump() for source in result.sources]))
        db.commit()
        return result
    except RuntimeError:
        db.rollback()
        raise HTTPException(503, "Question answering unavailable; check LLM_API_KEY, LLM_MODEL and LLM_BASE_URL") from None
    except Exception:
        db.rollback()
        raise HTTPException(503, "Question answering unavailable") from None


@router.get("/history", response_model=list[HistoryItem])
def history(limit: int = Query(default=20, ge=1), db: Session = Depends(get_db)):
    if limit > 100:
        raise HTTPException(422, "limit must be between 1 and 100")
    return db.scalars(select(QueryLog).order_by(QueryLog.created_at.desc(), QueryLog.id.desc()).limit(limit)).all()
