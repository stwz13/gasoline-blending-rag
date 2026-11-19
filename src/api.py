"""RAG-сервис: HTTP API, который отвечает на вопросы по регламенту АССБ.

Сервис при запуске один раз загружает поиск (гибрид BM25 и вектора с переранжированием), а на каждый
вопрос находит фрагменты регламента и получает ответ языковой модели через Ollama.

Запуск:
    python -m src.api

Описание API с формой для проверки запросов: http://localhost:8000/docs

Настройки поиска и Ollama хранятся в config.toml.
"""

import asyncio
import json
import time
from contextlib import aclosing, asynccontextmanager

import httpx
import requests
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.generate import DEFAULT_K, DEFAULT_MODEL, OLLAMA_BASE_URL, answer, build_messages, source_label, stream_ollama
from src.hybrid_retrieval import RerankedRetriever, build_hybrid, make_reranker
from src.retrieval import Retriever, build_retriever, default_bm25
from src.settings import settings

MODEL = DEFAULT_MODEL


class Question(BaseModel):
    """Запрос к сервису."""

    question: str = Field(min_length=1, max_length=1000, description="вопрос по регламенту")
    k: int = Field(DEFAULT_K, ge=1, le=settings.retrieval.max_k, description="число фрагментов в контексте модели")


class Source(BaseModel):
    """Фрагмент регламента, переданный модели."""

    number: int = Field(description="номер фрагмента, на который ссылается ответ: [1], [2], ...")
    chunk_id: str
    label: str = Field(description="документ и раздел")
    text: str


class Reply(BaseModel):
    """Ответ сервиса."""

    answer: str
    sources: list[Source]
    model: str
    seconds: float = Field(description="время генерации ответа моделью, с")


class Health(BaseModel):
    """Состояние сервиса и его зависимостей."""

    ollama: bool = Field(description="сервер Ollama доступен")
    model: str
    model_ready: bool = Field(description="модель скачана в Ollama")


state: dict = {}


class CancellableStream(StreamingResponse):
    """Закрывает генератор при отключении клиента, даже между токенами."""

    async def __call__(self, scope, receive, send):
        async with aclosing(self.body_iterator):
            stream = asyncio.create_task(self.stream_response(send))
            disconnect = asyncio.create_task(self.listen_for_disconnect(receive))
            try:
                ready, _ = await asyncio.wait({stream, disconnect}, return_when=asyncio.FIRST_COMPLETED)
                for task in ready:
                    task.result()
            finally:
                stream.cancel()
                disconnect.cancel()
                await asyncio.gather(stream, disconnect, return_exceptions=True)


def configured_retriever() -> Retriever:
    config = settings.retrieval
    if config.method in ("rrf", "weighted"):
        return build_hybrid(config.method, config.encoder, config.alpha, config.rrf_k, config.rerank,
                            bm25_field=config.bm25_field, rerank_depth=config.rerank_depth)
    if config.method == "bm25":
        retriever = default_bm25(field=config.bm25_field)
    else:
        retriever = build_retriever("dense", config.encoder)
    if config.rerank:
        return RerankedRetriever(retriever, make_reranker(), depth=config.rerank_depth)
    return retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = configured_retriever()
    state["lock"] = asyncio.Lock()
    yield
    state.clear()


app = FastAPI(
    title="RAG по регламенту АССБ",
    description="Ответы на вопросы по технологическому регламенту станции смешения бензинов со ссылками на фрагменты.",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health)
def health() -> Health:
    """Проверяет, доступна ли Ollama и скачана ли модель."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return Health(ollama=False, model=MODEL, model_ready=False)
    names = {m["name"] for m in response.json().get("models", [])}
    ready = MODEL in names or f"{MODEL}:latest" in names
    return Health(ollama=True, model=MODEL, model_ready=ready)


@app.post("/ask", response_model=Reply)
async def ask(query: Question) -> Reply:
    """Находит фрагменты регламента и возвращает ответ модели со ссылками на них."""
    num_ctx = settings.ollama.large_num_ctx if query.k > DEFAULT_K else settings.ollama.num_ctx
    async with state["lock"]:
        try:
            result = await run_in_threadpool(answer, query.question, state["retriever"], MODEL, query.k, num_ctx)
        except requests.ConnectionError as error:
            raise HTTPException(503, f"Сервер Ollama недоступен по адресу {OLLAMA_BASE_URL}") from error
        except requests.RequestException as error:
            raise HTTPException(502, f"Ошибка Ollama: {error}") from error
    sources = [
        Source(number=i, chunk_id=hit.chunk_id, label=source_label(hit), text=hit.chunk["text"])
        for i, hit in enumerate(result.sources, start=1)
    ]
    return Reply(answer=result.text, sources=sources, model=result.model, seconds=round(result.seconds, 1))


@app.post("/ask/stream")
def ask_stream(query: Question):
    """Отдает ответ в формате событий NDJSON, который ожидает app/chat.py."""
    num_ctx = settings.ollama.large_num_ctx if query.k > DEFAULT_K else settings.ollama.num_ctx

    async def event_generator():
        try:
            async with state["lock"]:
                start = time.perf_counter()
                search = asyncio.create_task(run_in_threadpool(state["retriever"].search, query.question, query.k))
                try:
                    hits = await asyncio.shield(search)
                except asyncio.CancelledError:
                    # Поиск в потоке должен освободить общие ресурсы до следующего запроса.
                    await asyncio.gather(search, return_exceptions=True)
                    raise
                retrieval_seconds = time.perf_counter() - start
                sources = [
                    Source(number=i, chunk_id=hit.chunk_id, label=source_label(hit), text=hit.chunk["text"]).model_dump()
                    for i, hit in enumerate(hits, start=1)
                ]
                yield json.dumps({"type": "sources", "sources": sources,
                                  "retrieval_seconds": round(retrieval_seconds, 2)}, ensure_ascii=False) + "\n"
                start = time.perf_counter()
                tokens = stream_ollama(build_messages(query.question, hits), MODEL, num_ctx)
                async with aclosing(tokens):
                    async for token in tokens:
                        yield json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"
                seconds = round(time.perf_counter() - start, 1)
        except httpx.ConnectError:
            detail = f"Сервер Ollama недоступен по адресу {OLLAMA_BASE_URL}"
        except (httpx.HTTPError, ValueError) as error:
            detail = f"Ошибка Ollama: {error}"
        else:
            yield json.dumps({"type": "done", "model": MODEL, "seconds": seconds}, ensure_ascii=False) + "\n"
            return
        yield json.dumps({"type": "error", "detail": detail}, ensure_ascii=False) + "\n"

    return CancellableStream(event_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api.host, port=settings.api.port)
