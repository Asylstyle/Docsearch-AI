from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from pypdf.errors import PdfReadError

from app.db import get_db
from app.ingestion.service import ingest_document
from app.models import Document
from app.search.service import invalidate_search_index

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=201, summary="Upload and index a text PDF")
def upload_document(file: UploadFile = File(..., description="PDF document"), db: Session = Depends(get_db)):
    raw = file.file.read(10 * 1024 * 1024 + 1)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "PDF exceeds 10 MiB limit")
    if not raw.startswith(b"%PDF-"):
        raise HTTPException(422, "Invalid PDF signature")
    filename = (file.filename or "document.pdf").replace("\\", "/").rsplit("/", 1)[-1][:255]
    try:
        return ingest_document(db, filename, raw)
    except (ValueError, PdfReadError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        # Failed Document remains inspectable; never expose exception credentials.
        raise HTTPException(503, "Ingestion failed; inspect document status") from exc


@router.get("", summary="List documents and ingestion status")
def list_documents(db: Session = Depends(get_db)):
    return db.scalars(select(Document).order_by(Document.id)).all()


@router.delete("/{id}", status_code=204, summary="Delete a document and its chunks")
def delete_document(id: int, db: Session = Depends(get_db)):
    document = db.get(Document, id)
    if document is None:
        raise HTTPException(404, "Document not found")
    db.delete(document)
    db.commit()
    invalidate_search_index()
    return Response(status_code=204)
