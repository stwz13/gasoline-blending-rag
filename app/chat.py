"""Веб-интерфейс: чат с RAG-сервисом по регламенту АССБ.

Интерфейс не ищет фрагменты и не запускает модель сам: он отправляет вопрос RAG-сервису (src/api.py)
и показывает ответ, а рядом — фрагменты регламента, на которые ссылается ответ.

Запуск:
    python app/chat.py               # http://localhost:7860
    python app/chat.py --share       # плюс временная публичная ссылка через серверы Gradio

Переменная окружения RAG_API_URL — адрес RAG-сервиса, по умолчанию http://localhost:8000.
"""

import argparse
import json
import os
from collections.abc import Iterator

import gradio as gr
import requests

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 900
WAITING = "⏳ Ищу фрагменты регламента…"
WAITING_MODEL = "⏳ Фрагменты найдены, модель готовит ответ. На процессоре это может занять 1–2 минуты…"
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


def add_question(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Сразу показывает вопрос в чате, пока модель готовит ответ."""
    if not question.strip():
        return "", history
    return "", history + [{"role": "user", "content": question.strip()}]


def get_answer(history: list[dict]) -> Iterator[tuple[list[dict], str]]:
    """Запрашивает ответ у RAG-сервиса на последний вопрос и показывает его по мере генерации.

    Источники приходят первыми и сразу появляются справа, затем текст ответа дописывается по словам.
    """
    if not history or history[-1]["role"] != "user":
        yield history, gr.skip()
        return
    question = message_text(history[-1])
    history = history + [{"role": "assistant", "content": WAITING}]
    sources = NO_SOURCES
    yield history, sources
    reply = ""
    try:
        with requests.post(f"{API_URL}/ask/stream", json={"question": question},
                           stream=True, timeout=TIMEOUT) as response:
            if not response.ok:
                history[-1]["content"] = f"⚠️ {response.json().get('detail', response.text)}"
                yield history, sources
                return
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "sources":
                    sources = format_sources(event["sources"])
                    history[-1]["content"] = WAITING_MODEL
                elif event["type"] == "token":
                    reply += event["text"]
                    history[-1]["content"] = reply
                elif event["type"] == "done":
                    history[-1]["content"] = f"{reply}\n\n<sub>{event['model']}, {event['seconds']:.1f} с</sub>"
                elif event["type"] == "error":
                    history[-1]["content"] = f"{reply}\n\n⚠️ {event['detail']}".strip()
                yield history, sources
    except requests.RequestException:
        history[-1]["content"] = f"⚠️ RAG-сервис недоступен по адресу {API_URL}."
        yield history, sources


def build_ui() -> gr.Blocks:
    """Собирает интерфейс: чат слева, источники и настройки справа."""
    with gr.Blocks(title="Регламент АССБ") as demo:
        gr.Markdown("# Помощник технолога АССБ\n"
                    "Ответы по технологическому регламенту станции смешения бензинов. Модель отвечает только "
                    "по найденным фрагментам регламента и ставит ссылки на них вида [1].")
        status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Чат", height=520)
                question = gr.Textbox(placeholder="Задайте вопрос по регламенту", show_label=False,
                                      submit_btn="Спросить")
                gr.Examples(EXAMPLES, inputs=question, label="Примеры вопросов")
            with gr.Column(scale=2):
                clear = gr.Button("Очистить чат")
                gr.Markdown("### Источники последнего ответа")
                sources = gr.Markdown(NO_SOURCES)

        question.submit(add_question, [question, chatbot], [question, chatbot], queue=False).then(
            get_answer, chatbot, [chatbot, sources])
        clear.click(lambda: ([], NO_SOURCES), None, [chatbot, sources], queue=False)
        demo.load(check_service, None, status)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Веб-интерфейс чата по регламенту АССБ")
    parser.add_argument("--host", default="127.0.0.1", help="адрес, на котором слушает интерфейс")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="создать публичную ссылку Gradio")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
