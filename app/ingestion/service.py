from app.models import Document
from app.schemas import ChunkData


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    raise NotImplementedError


def split_into_chunks(pages) -> list[ChunkData]:
    raise NotImplementedError


def ingest_document(db, filename, file_bytes) -> Document:
    raise NotImplementedError
