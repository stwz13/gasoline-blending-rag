"""RAG-сервис: HTTP API, который отвечает на вопросы по регламенту АССБ.

Сервис при запуске один раз загружает поиск (гибрид BM25 и вектора с переранжированием), а на каждый
вопрос находит фрагменты регламента и получает ответ языковой модели через Ollama.

Запуск:
    uvicorn src.api:app --port 8000

Описание API с формой для проверки запросов: http://localhost:8000/docs

Переменные окружения:
    OLLAMA_BASE_URL — адрес сервера Ollama, по умолчанию http://localhost:11434;
    RAG_MODEL — имя модели в Ollama, по умолчанию YandexGPT 5 Lite 8B.
"""

import os
import threading
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generate import DEFAULT_K, DEFAULT_MODEL, OLLAMA_BASE_URL, answer, source_label
from src.hybrid_retrieval import build_hybrid

MODEL = os.environ.get("RAG_MODEL", DEFAULT_MODEL)
MAX_K = 10


class Question(BaseModel):
    """Запрос к сервису."""

    question: str = Field(min_length=1, max_length=1000, description="вопрос по регламенту")
    k: int = Field(DEFAULT_K, ge=1, le=MAX_K, description="число фрагментов в контексте модели")


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
# Модель в Ollama обрабатывает запросы по одному, поэтому вопросы ставятся в очередь
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = build_hybrid("rrf", rerank=True)
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
def ask(query: Question) -> Reply:
    """Находит фрагменты регламента и возвращает ответ модели со ссылками на них."""
    num_ctx = 16384 if query.k > DEFAULT_K else 8192
    with lock:
        try:
            result = answer(query.question, state["retriever"], MODEL, query.k, num_ctx)
        except requests.ConnectionError as error:
            raise HTTPException(503, f"Сервер Ollama недоступен по адресу {OLLAMA_BASE_URL}") from error
        except requests.RequestException as error:
            raise HTTPException(502, f"Ошибка Ollama: {error}") from error
    sources = [
        Source(number=i, chunk_id=hit.chunk_id, label=source_label(hit), text=hit.chunk["text"])
        for i, hit in enumerate(result.sources, start=1)
    ]
    return Reply(answer=result.text, sources=sources, model=result.model, seconds=round(result.seconds, 1))
