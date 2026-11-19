"""Общие настройки API, генерации и чата."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalSettings(ConfigSection):
    method: Literal["bm25", "dense", "rrf", "weighted"]
    encoder: str
    bm25_field: Literal["text", "embed_text"]
    rerank: bool
    rerank_depth: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    alpha: float = Field(ge=0, le=1)
    default_k: int = Field(gt=0)
    max_k: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_depth(self):
        if self.default_k > self.max_k:
            raise ValueError("default_k должен быть не больше max_k")
        return self


class OllamaSettings(ConfigSection):
    base_url: str
    model: str
    temperature: float = Field(ge=0)
    num_ctx: int = Field(gt=0)
    large_num_ctx: int = Field(gt=0)
    timeout: float = Field(gt=0)
    num_thread: int | None = Field(default=None, gt=0)
    num_predict: int | None = Field(default=None, gt=0)
    keep_alive: int | str | None = None


class ServerSettings(ConfigSection):
    host: str
    port: int = Field(ge=1, le=65535)


class ChatSettings(ServerSettings):
    api_url: str
    share: bool
    timeout: float = Field(gt=0)
    animation_interval: float = Field(gt=0)


class Settings(ConfigSection):
    retrieval: RetrievalSettings
    ollama: OllamaSettings
    api: ServerSettings
    chat: ChatSettings


def load_settings() -> Settings:
    path = Path(os.environ.get("RAG_CONFIG", Path(__file__).resolve().parent.parent / "config.toml"))
    with path.open("rb") as file:
        data = tomllib.load(file)
    profiles = data.pop("profiles", {})
    profile = os.environ.get("RAG_PROFILE", "default")
    if profile != "default":
        if profile not in profiles:
            raise ValueError(f"Профиль {profile!r} отсутствует в {path}")
        for section, overrides in profiles[profile].items():
            data[section] = {**data.get(section, {}), **overrides}

    for env, section, field in (
        ("OLLAMA_BASE_URL", "ollama", "base_url"),
        ("RAG_MODEL", "ollama", "model"),
        ("RAG_API_URL", "chat", "api_url"),
    ):
        if env in os.environ:
            data[section][field] = os.environ[env]
    return Settings.model_validate(data)


settings = load_settings()
