"""Small contract checks: no database, model download, network or live LLM."""
from io import BytesIO
import socket
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.config import settings
from app.db import get_db
from app.ingestion.service import extract_pages, split_into_chunks
from app.main import app
import app.rag.service as rag
from app.schemas import SearchHit


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network is forbidden in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setenv('HF_HUB_OFFLINE','1')


def test_chunk_size_overlap_global_index_and_page_boundaries(monkeypatch):
    monkeypatch.setattr(settings,'CHUNK_SIZE',5)
    monkeypatch.setattr(settings,'CHUNK_OVERLAP',2)
    chunks=split_into_chunks([(1,'abcdefghij'),(3,'KLMNOPQ')])
    assert [(c.page,c.chunk_index,c.text) for c in chunks] == [
        (1,0,'abcde'),(1,1,'defgh'),(1,2,'ghij'),(3,3,'KLMNO'),(3,4,'NOPQ')]
    assert all(len(c.text)<=5 for c in chunks)
    assert chunks[0].text[-2:]==chunks[1].text[:2]


def test_invalid_stride_cannot_loop(monkeypatch):
    for size,overlap in [(0,0),(5,5),(5,-1)]:
        monkeypatch.setattr(settings,'CHUNK_SIZE',size)
        monkeypatch.setattr(settings,'CHUNK_OVERLAP',overlap)
        with pytest.raises(ValueError):
            split_into_chunks([(1,'abcdefghij')])


def test_extract_original_one_based_pages_skips_blank():
    writer=PdfWriter()
    for text in ['alpha',None,'gamma']:
        page=writer.add_blank_page(width=200,height=200)
        if text:
            font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),
                NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
            page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):
                DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
            stream=DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 10 100 Td ({text}) Tj ET'.encode())
            page[NameObject('/Contents')]=writer._add_object(stream)
    output=BytesIO(); writer.write(output)
    assert extract_pages(output.getvalue())==[(1,'alpha'),(3,'gamma')]


def stub_llm(monkeypatch, answer):
    hits=[SearchHit(chunk_id=i,document_id=10 if i<3 else 11,
        filename='a.pdf' if i<3 else 'b.pdf',page=3 if i<3 else 4,text='offline fixture',score=.8)
        for i in (1,2,3)]
    monkeypatch.setattr(rag,'search_chunks',lambda *args: hits)
    monkeypatch.setattr(settings,'LLM_API_KEY','offline-placeholder-not-a-key')
    class Client:
        def __init__(self,**kwargs):
            self.chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
                SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=answer))])))
        def __enter__(self): return self
        def __exit__(self,*args): return False
    monkeypatch.setattr(rag,'OpenAI',Client)


def test_rag_deduplicates_document_page_and_ignores_invalid_numbers(monkeypatch):
    stub_llm(monkeypatch,'Первое [1]. То же [2][1]. Ошибка [0][999]. Второе [3].')
    result=rag.answer_question(None,'Вопрос')
    assert [(s.document_id,s.page) for s in result.sources]==[(10,3),(11,4)]


def test_rag_answer_without_valid_references_has_empty_sources(monkeypatch):
    stub_llm(monkeypatch,'Утверждение без валидных ссылок [999].')
    result=rag.answer_question(None,'Вопрос')
    assert result.answer=='Утверждение без валидных ссылок [999].'
    assert result.sources==[]


def test_rag_refusal_has_no_sources(monkeypatch):
    stub_llm(monkeypatch,rag.REFUSAL+' [1]')
    result=rag.answer_question(None,'Вопрос')
    assert result.answer==rag.REFUSAL and result.sources==[]


def test_health():
    with TestClient(app) as client:
        response=client.get('/health')
    assert response.status_code==200
    assert response.json()=={'status':'ok'}


def test_ask_empty_retrieval_refuses_without_llm(monkeypatch):
    monkeypatch.setattr(rag,'search_chunks',lambda *args: [])
    def forbidden(**kwargs): raise AssertionError('LLM must not be called')
    monkeypatch.setattr(rag,'OpenAI',forbidden)
    saved=[]
    db=SimpleNamespace(add=saved.append,commit=lambda:None,rollback=lambda:None)
    app.dependency_overrides[get_db]=lambda:db
    try:
        with TestClient(app) as client:
            response=client.post('/ask',json={'question':'Вопрос без контекста','k':5})
        assert response.status_code==200
        assert response.json()=={'answer':rag.REFUSAL,'sources':[]}
        assert len(saved)==1 and saved[0].query_type=='ask'
    finally:
        app.dependency_overrides.clear()
