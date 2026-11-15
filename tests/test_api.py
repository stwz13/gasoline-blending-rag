"""Проверки HTTP API RAG-сервиса без настоящего поиска и модели."""

import pytest
import requests
from fastapi.testclient import TestClient

import src.api as api
import src.generate as generate
from src.retrieval import Hit

CHUNK = {"chunk_id": "09-003", "doc_title": "Правила закачки компонентов", "section": "4. Перемешивание",
         "text": "После закачки всех компонентов бензин перемешивают не менее 2 часов."}


class FakeRetriever:
    def search(self, query: str, k: int) -> list[Hit]:
        return [Hit(CHUNK["chunk_id"], 1.0, 1, CHUNK)][:k]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "build_hybrid", lambda *args, **kwargs: FakeRetriever())
    with TestClient(api.app) as client:
        yield client


def test_ask_returns_answer_and_sources(client, monkeypatch):
    calls = []

    def fake_ollama(messages, model, num_ctx):
        calls.append((model, num_ctx))
        return "Не менее 2 часов [1]."

    monkeypatch.setattr(generate, "ask_ollama", fake_ollama)
    response = client.post("/ask", json={"question": "Сколько перемешивать бензин?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Не менее 2 часов [1]."
    assert data["sources"][0]["number"] == 1
    assert data["sources"][0]["chunk_id"] == "09-003"
    assert "Правила закачки компонентов" in data["sources"][0]["label"]
    assert calls == [(api.MODEL, 8192)]


def test_ask_uses_larger_context_for_10_chunks(client, monkeypatch):
    calls = []
    monkeypatch.setattr(generate, "ask_ollama", lambda messages, model, num_ctx: calls.append(num_ctx) or "ответ")
    assert client.post("/ask", json={"question": "вопрос", "k": 10}).status_code == 200
    assert calls == [16384]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "вопрос", "k": 50}).status_code == 422


def test_ask_reports_unavailable_ollama(client, monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError

    monkeypatch.setattr(generate, "ask_ollama", down)
    response = client.post("/ask", json={"question": "вопрос"})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_health_without_ollama(client, monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError

    monkeypatch.setattr(api.requests, "get", down)
    assert client.get("/health").json() == {"ollama": False, "model": api.MODEL, "model_ready": False}
