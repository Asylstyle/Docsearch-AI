from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import settings
from app.models import Chunk
from app.models import Document
from app.schemas import ChunkData
from app.search.service import embed_texts, invalidate_search_index


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((number, text))
        return pages
    except PdfReadError as exc:
        raise ValueError("Invalid PDF") from exc


def split_into_chunks(pages) -> list[ChunkData]:
    size, overlap = settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require CHUNK_SIZE > 0 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
    chunks = []
    for page, text in pages:
        for start in range(0, len(text), size - overlap):
            part = text[start:start + size]
            if part.strip():
                chunks.append(ChunkData(page=page, chunk_index=len(chunks), text=part))
            if start + size >= len(text):
                break
    return chunks


def ingest_document(db, filename, file_bytes) -> Document:
    document = Document(filename=filename, status="processing")
    db.add(document)
    # Persist the record first: rollback of chunks must not erase failure status.
    db.commit()
    document_id = document.id
    pages_count = 0
    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        # Includes blank pages, including trailing blanks skipped by extract_pages.
        pages_count = len(reader.pages)
        pages = extract_pages(file_bytes)
        chunks = split_into_chunks(pages)
        if not chunks:
            raise ValueError("No extractable text; scanned PDFs require OCR")
        embeddings = embed_texts([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")
        db.add_all([Chunk(document_id=document_id, page=chunk.page,
                          chunk_index=chunk.chunk_index, text=chunk.text, embedding=embedding)
                    for chunk, embedding in zip(chunks, embeddings)])
        document.pages_count = pages_count
        document.status = "ready"
        db.commit()
        invalidate_search_index()
        return document
    except Exception:
        db.rollback()
        failed = db.get(Document, document_id)
        failed.pages_count = pages_count
        failed.status = "failed"
        db.commit()
        raise
