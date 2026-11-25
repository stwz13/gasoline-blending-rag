"""Генерация ответа локальной LLM по найденным фрагментам регламента.

Модель запускается через Ollama (https://ollama.com): сервер Ollama должен быть запущен,
а модель скачана командой `ollama pull <модель>`. Адрес сервера Ollama задаётся
переменной окружения OLLAMA_BASE_URL, по умолчанию http://localhost:11434.
"""

import argparse
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import requests

from src.hybrid_retrieval import build_hybrid
from src.retrieval import Hit, Retriever
from src.settings import settings

OLLAMA_BASE_URL = settings.ollama.base_url.rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
DEFAULT_MODEL = settings.ollama.model
DEFAULT_K = settings.retrieval.default_k
NOT_FOUND = "Информация не найдена в регламенте."

INSTRUCTIONS = f"""Ты — помощник технолога автоматизированной станции смешения бензинов (АССБ).
Отвечай на вопрос только по фрагментам регламента, приведённым ниже.
Вопрос может быть сформулирован другими словами, чем текст регламента: ищи фрагменты, близкие по смыслу.
Перед ответом определи, что именно спрашивают, и найди во фрагменте подтверждение именно этого факта.
Упоминание той же установки или процесса само по себе не отвечает на вопрос. Например, если спрашивают
рабочее давление, сведения о производительности оборудования не подтверждают ответ.
Не добавляй факты, которых нет во фрагментах, даже если знаешь их из общих знаний.
После каждого утверждения укажи номер фрагмента в квадратных скобках, например [2].
Если ни один фрагмент не подтверждает запрошенный факт, ответь ровно: «{NOT_FOUND}»
Отвечай по-русски, кратко и по существу."""


@dataclass
class Answer:
    """Ответ модели.

    Attributes:
        question: вопрос пользователя.
        text: ответ модели.
        sources: найденные фрагменты в том порядке, в котором они переданы модели.
        model: имя модели.
        seconds: время генерации в секундах.
    """

    question: str
    text: str
    sources: list[Hit]
    model: str
    seconds: float


def source_label(hit: Hit) -> str:
    """Подписывает фрагмент для ссылки: документ и раздел.

    Args:
        hit: найденный фрагмент.

    Returns:
        Строка вида «Название документа, раздел (чанк 03-005)».
    """
    chunk = hit.chunk
    section = chunk.get("subsection") or chunk.get("section") or ""
    return f"{chunk['doc_title']}, {section} (чанк {hit.chunk_id})"


def build_context(hits: list[Hit]) -> str:
    """Собирает пронумерованные фрагменты для промпта.

    Args:
        hits: найденные фрагменты.

    Returns:
        Фрагменты с номерами [1], [2], ... и подписями источников.
    """
    return "\n\n".join(f"[{i}] {source_label(hit)}\n{hit.chunk['text']}" for i, hit in enumerate(hits, start=1))


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    """Собирает сообщение для чат-модели.

    Инструкции передаются в начале сообщения пользователя, а не отдельным системным сообщением:
    шаблон чата YandexGPT 5 Lite в Ollama отбрасывает роль system, и модель их не видит.

    Args:
        question: вопрос пользователя.
        hits: найденные фрагменты.

    Returns:
        Список из одного сообщения с ролью user: инструкции, фрагменты и вопрос.
    """
    user = f"{INSTRUCTIONS}\n\nФрагменты регламента:\n\n{build_context(hits)}\n\nВопрос: {question}"
    return [{"role": "user", "content": user}]


def ask_ollama(messages: list[dict], model: str = DEFAULT_MODEL, num_ctx: int = settings.ollama.num_ctx,
               url: str = OLLAMA_URL, timeout: float = settings.ollama.timeout) -> str:
    """Отправляет сообщения локальной модели через Ollama.

    Args:
        messages: сообщения чата.
        model: имя модели в Ollama.
        num_ctx: размер контекстного окна в токенах; для 10 фрагментов нужно больше 8192.
        url: адрес API чата Ollama.
        timeout: предельное время ожидания ответа в секундах.

    Returns:
        Текст ответа модели.

    Raises:
        requests.HTTPError: Ollama вернула ошибку, например модель не скачана.
    """
    with ollama_request(messages, model, num_ctx, url, timeout, stream=False) as response:
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


def ollama_request(messages: list[dict], model: str, num_ctx: int, url: str,
                   timeout: float, *, stream: bool) -> requests.Response:
    payload = ollama_payload(messages, model, num_ctx, stream=stream)
    return requests.post(url, timeout=timeout, stream=stream, json=payload)


def ollama_payload(messages: list[dict], model: str, num_ctx: int, *, stream: bool) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": settings.ollama.temperature, "num_ctx": num_ctx},
    }
    if settings.ollama.num_thread is not None:
        payload["options"]["num_thread"] = settings.ollama.num_thread
    if settings.ollama.num_predict is not None:
        payload["options"]["num_predict"] = settings.ollama.num_predict
    if settings.ollama.keep_alive is not None:
        payload["keep_alive"] = settings.ollama.keep_alive
    return payload


async def stream_ollama(messages: list[dict], model: str, num_ctx: int,
                        url: str = OLLAMA_URL, timeout: float = settings.ollama.timeout) -> AsyncIterator[str]:
    payload = ollama_payload(messages, model, num_ctx, stream=True)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise httpx.RequestError(event["error"])
                content = event.get("message", {}).get("content")
                if content:
                    yield content
                if event.get("done"):
                    return
    raise httpx.RequestError("Ollama завершила поток без финального события")


def answer(question: str, retriever: Retriever, model: str = DEFAULT_MODEL, k: int = DEFAULT_K,
           num_ctx: int = settings.ollama.num_ctx) -> Answer:
    """Находит фрагменты и генерирует ответ со ссылками на них.

    Args:
        question: вопрос пользователя.
        retriever: метод поиска.
        model: имя модели в Ollama.
        k: число фрагментов в контексте.
        num_ctx: размер контекстного окна модели в токенах.

    Returns:
        Ответ модели и использованные фрагменты.
    """
    hits = retriever.search(question, k)
    start = time.perf_counter()
    text = ask_ollama(build_messages(question, hits), model, num_ctx)
    return Answer(question, text, hits, model, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ответ на вопрос по регламенту АССБ")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели в Ollama")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help="число фрагментов в контексте")
    args = parser.parse_args()

    retriever = build_hybrid("rrf", rerank=True)
    result = answer(args.question, retriever, args.model, args.k)
    print(result.text)
    print("\nИсточники:")
    for i, hit in enumerate(result.sources, start=1):
        print(f"[{i}] {source_label(hit)}")
    print(f"\n{result.model}, {result.seconds:.1f} с")


if __name__ == "__main__":
    main()
