from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", summary="Upload a PDF (not implemented)", responses={501: {"description": "Scaffold only"}})
def upload_document(file: UploadFile = File(..., description="PDF document"), db: Session = Depends(get_db)):
    raise NotImplementedError


@router.get("", summary="List documents (not implemented)", responses={501: {"description": "Scaffold only"}})
def list_documents(db: Session = Depends(get_db)):
    raise NotImplementedError


@router.delete("/{id}", summary="Delete a document (not implemented)", responses={501: {"description": "Scaffold only"}})
def delete_document(id: int, db: Session = Depends(get_db)):
    raise NotImplementedError
