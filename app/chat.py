"""Веб-интерфейс: чат с RAG-сервисом по регламенту АССБ.

Интерфейс не ищет фрагменты и не запускает модель сам: он отправляет вопрос RAG-сервису (src/api.py)
и показывает ответ, а рядом — фрагменты регламента, на которые ссылается ответ.

Запуск:
    python app/chat.py               # http://localhost:7860
    python app/chat.py --share       # плюс временная публичная ссылка через серверы Gradio

Настройки подключения и запуска интерфейса хранятся в config.toml.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import httpx
import requests

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.settings import settings

API_URL = settings.chat.api_url.rstrip("/")
WAITING = "⏳ Ищу фрагменты регламента"
WAITING_MODEL = "⏳ Готовлю ответ"
DOTS = ("...", "..", ".")
NO_SOURCES = "*Здесь появятся фрагменты регламента, по которым составлен ответ.*"
EXAMPLES = [
    "Сколько времени перемешивать бензин после закачки компонентов?",
    "Какие компоненты входят в рецептуру АИ-95?",
    "Почему октановое число смеси не равно средневзвешенному?",
    "Как уберечь оборудование от искр статики?",
    "Что такое ДНП?",
]


def check_service() -> str:
    """Возвращает строку состояния RAG-сервиса и модели."""
    try:
        status = requests.get(f"{API_URL}/health", timeout=10).json()
    except requests.RequestException:
        return f"🔴 RAG-сервис недоступен по адресу {API_URL}"
    if not status["ollama"]:
        return "🔴 RAG-сервис работает, но сервер Ollama недоступен"
    if not status["model_ready"]:
        return f"🟡 Модель не скачана: `ollama pull {status['model']}`"
    return f"🟢 Сервис готов, модель `{status['model']}`"


def format_sources(sources: list[dict]) -> str:
    """Оформляет фрагменты-источники в Markdown."""
    if not sources:
        return NO_SOURCES
    parts = []
    for source in sources:
        text = source["text"].strip()
        parts.append(f"**[{source['number']}] {source['label']}**\n\n<details><summary>Текст фрагмента</summary>\n\n"
                     f"{text}\n\n</details>")
    return "\n\n".join(parts)


def message_text(message: dict) -> str:
    """Достаёт текст сообщения чата: Gradio хранит его строкой или списком частей."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


@dataclass
class Generation:
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    clear: bool = False
    started: bool = False


async def add_question(question: str, history: list[dict], generation: Generation | None):
    """Сразу показывает вопрос в чате, пока модель готовит ответ."""
    if generation is not None or not question.strip():
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    history = history + [{"role": "user", "content": question.strip()}]
    return gr.update(value="", interactive=False), history, Generation(), gr.update(interactive=True)


async def stop_answer(generation: Generation | None):
    if generation is not None:
        generation.stop.set()


async def clear_chat(generation: Generation | None):
    if generation is not None:
        generation.clear = True
        generation.stop.set()
        return gr.skip(), gr.skip()
    return [], NO_SOURCES


async def answer_events(question: str) -> AsyncIterator[dict]:
    async with httpx.AsyncClient(timeout=settings.chat.timeout) as client:
        async with client.stream("POST", f"{API_URL}/ask/stream", json={"question": question}) as response:
            if not response.is_success:
                await response.aread()
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                yield {"type": "error", "detail": detail}
                return
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)


async def get_answer(history: list[dict], generation: Generation | None) -> AsyncIterator[tuple]:
    """Показывает ответ и обновляет многоточие между событиями API."""
    if generation is None or generation.started or not history or history[-1]["role"] != "user":
        return
    generation.started = True
    question = message_text(history[-1])
    sources = NO_SOURCES
    reply = ""
    status = WAITING
    interval = settings.chat.animation_interval
    loop = asyncio.get_running_loop()
    start = loop.time()

    def render(content: str):
        return history + [{"role": "assistant", "content": content}]

    events = answer_events(question)
    pending = asyncio.create_task(anext(events, None))
    stopped = asyncio.create_task(generation.stop.wait())
    try:
        while True:
            if generation.stop.is_set():
                final = f"{reply.strip()}\n\n*Ответ остановлен.*".strip()
                break
            frame = int((loop.time() - start) / interval)
            yield render(f"{reply.rstrip() or status} {DOTS[frame % len(DOTS)]}"), sources, gr.skip(), gr.skip(), gr.skip()
            next_frame = start + (frame + 1) * interval
            ready, _ = await asyncio.wait({pending, stopped}, timeout=max(0, next_frame - loop.time()),
                                          return_when=asyncio.FIRST_COMPLETED)
            if stopped in ready or pending not in ready:
                continue
            event = pending.result()
            if event is None:
                final = f"{reply}\n\n⚠️ API завершил поток без финального события.".strip()
                break
            if event["type"] == "done":
                final = f"{reply.strip()}\n\n<sub>{event['model']}, {event['seconds']:.1f} с</sub>"
                break
            if event["type"] == "error":
                final = f"{reply}\n\n⚠️ {event['detail']}".strip()
                break
            if event["type"] == "sources":
                sources = format_sources(event["sources"])
                status = WAITING_MODEL
            elif event["type"] == "token":
                reply += event["text"]
            pending = asyncio.create_task(anext(events, None))
    except (httpx.RequestError, ValueError):
        final = f"{reply}\n\n⚠️ Не удалось дочитать ответ от RAG-сервиса ({API_URL}).".strip()
    finally:
        pending.cancel()
        stopped.cancel()
        await asyncio.gather(pending, stopped, return_exceptions=True)
        await events.aclose()
    messages = [] if generation.clear else render(final)
    yield messages, NO_SOURCES if generation.clear else sources, None, gr.update(interactive=True), gr.update(interactive=False)


def build_ui() -> gr.Blocks:
    """Собирает интерфейс: чат слева, источники и настройки справа."""
    with gr.Blocks(title="Регламент АССБ") as demo:
        gr.Markdown("# Помощник технолога АССБ\n"
                    "Ответы по технологическому регламенту станции смешения бензинов. Модель отвечает только "
                    "по найденным фрагментам регламента и ставит ссылки на них вида [1].")
        status = gr.Markdown()
        generation = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Чат", height=520)
                question = gr.Textbox(placeholder="Задайте вопрос по регламенту", show_label=False,
                                      submit_btn="Спросить")
                stop = gr.Button("Остановить", interactive=False)
                gr.Examples(EXAMPLES, inputs=question, label="Примеры вопросов")
            with gr.Column(scale=2):
                clear = gr.Button("Очистить чат")
                gr.Markdown("### Источники последнего ответа")
                sources = gr.Markdown(NO_SOURCES)

        question.submit(add_question, [question, chatbot, generation], [question, chatbot, generation, stop],
                        queue=False).success(get_answer, [chatbot, generation],
                                             [chatbot, sources, generation, question, stop], concurrency_limit=None)
        stop.click(stop_answer, generation, None, queue=False)
        clear.click(clear_chat, generation, [chatbot, sources], queue=False)
        demo.load(check_service, None, status)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Веб-интерфейс чата по регламенту АССБ")
    parser.add_argument("--host", default=settings.chat.host, help="адрес, на котором слушает интерфейс")
    parser.add_argument("--port", type=int, default=settings.chat.port)
    parser.add_argument("--share", action="store_true", help="создать публичную ссылку Gradio")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=True if args.share or settings.chat.share else None)


if __name__ == "__main__":
    main()
