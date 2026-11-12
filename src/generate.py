"""Генерация ответа локальной LLM по найденным фрагментам регламента.

Модель запускается через Ollama (https://ollama.com): сервер Ollama должен быть запущен,
а модель скачана командой `ollama pull <модель>`.
"""

import argparse
import time
from dataclasses import dataclass

import requests

from src.hybrid_retrieval import build_hybrid
from src.retrieval import Hit, Retriever

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "hf.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF"
DEFAULT_K = 5
NOT_FOUND = "Информация не найдена в регламенте."

INSTRUCTIONS = f"""Ты — помощник технолога автоматизированной станции смешения бензинов (АССБ).
Отвечай на вопрос только по фрагментам регламента, приведённым ниже.
Вопрос может быть сформулирован другими словами, чем текст регламента: ищи фрагменты, близкие по смыслу.
Не добавляй факты, которых нет во фрагментах, даже если знаешь их из общих знаний.
После каждого утверждения укажи номер фрагмента в квадратных скобках, например [2].
Если во фрагментах нет ответа на вопрос, ответь ровно: «{NOT_FOUND}»
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


def ask_ollama(messages: list[dict], model: str = DEFAULT_MODEL, num_ctx: int = 8192,
               url: str = OLLAMA_URL, timeout: float = 900) -> str:
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
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def answer(question: str, retriever: Retriever, model: str = DEFAULT_MODEL, k: int = DEFAULT_K,
           num_ctx: int = 8192) -> Answer:
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
