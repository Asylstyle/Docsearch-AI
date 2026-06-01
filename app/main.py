from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import ask, documents, search

app = FastAPI(title="PDF RAG scaffold", version="0.1.0")
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)


@app.exception_handler(NotImplementedError)
async def not_implemented(request: Request, exc: NotImplementedError):
    return JSONResponse(status_code=501, content={"detail": "Not implemented: scaffold only"})


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
