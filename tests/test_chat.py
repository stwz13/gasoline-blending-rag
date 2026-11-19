import asyncio

from app import chat


def test_stop_keeps_partial_answer_and_closes_api_stream(monkeypatch):
    async def check():
        closed = asyncio.Event()

        async def fake_events(question):
            try:
                yield {"type": "token", "text": "Часть ответа"}
                await asyncio.Future()
            finally:
                closed.set()

        monkeypatch.setattr(chat, "answer_events", fake_events)
        generation = chat.Generation()
        history = [{"role": "user", "content": "Вопрос"}]
        async for messages, _, current, _, stop_button in chat.get_answer(history, generation):
            if current is None:
                assert messages[-1]["content"] == "Часть ответа\n\n*Ответ остановлен.*"
                assert not stop_button["interactive"]
                assert closed.is_set()
                return
            if messages[-1]["content"].startswith("Часть ответа"):
                await chat.stop_answer(generation)
        raise AssertionError("Чат не завершил остановленный ответ")

    asyncio.run(asyncio.wait_for(check(), 2))
