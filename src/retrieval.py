"""Поиск по чанкам корпуса АССБ: BM25 и векторный поиск."""

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import nltk
import numpy as np
import pymorphy3
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_DIR = DATA_DIR / "index"

GLOSSARY_DOC_ID = 13

ENCODERS = {
    "e5": "intfloat/multilingual-e5-small",
    "rubert-tiny2": "cointegrated/rubert-tiny2",
}

TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
GLOSSARY_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")

MORPH = pymorphy3.MorphAnalyzer()


def load_stopwords() -> set[str]:
    """Загружает список стоп-слов русского языка из NLTK.

    При первом запуске список скачивается командой nltk.download("stopwords").

    Returns:
        Стоп-слова в нижнем регистре, «ё» заменена на «е».
    """
    try:
        words = stopwords.words("russian")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("russian")
    return {word.replace("ё", "е") for word in words}


STOPWORDS = load_stopwords()


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Читает чанки из JSONL.

    Args:
        path: путь к файлу чанков.

    Returns:
        Список чанков в порядке файла.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    """Приводит слово к начальной форме.

    Args:
        word: слово в нижнем регистре.

    Returns:
        Лемма для русских слов, исходная строка для чисел и латиницы.
    """
    if not re.search("[а-я]", word):
        return word
    return MORPH.parse(word)[0].normal_form.replace("ё", "е")


def tokenize(text: str, lemmatize: bool = True) -> list[str]:
    """Разбивает текст на токены для BM25.

    Args:
        text: исходный текст.
        lemmatize: приводить ли слова к леммам.

    Returns:
        Токены в нижнем регистре без стоп-слов, «ё» заменена на «е».
    """
    words = TOKEN_RE.findall(text.lower().replace("ё", "е"))
    tokens = [lemma(w) if lemmatize else w for w in words]
    return [t for t in tokens if t not in STOPWORDS]


def load_glossary(chunks: list[dict]) -> dict[str, str]:
    """Собирает словарь сокращений из таблицы глоссария.

    Args:
        chunks: чанки корпуса.

    Returns:
        Словарь «сокращение → полная форма».
    """
    glossary = {}
    for chunk in chunks:
        if chunk.get("doc_id") != GLOSSARY_DOC_ID or chunk.get("chunk_type") != "table":
            continue
        for line in chunk["text"].splitlines():
            match = GLOSSARY_ROW_RE.match(line)
            if not match or set(match.group(1)) <= set("-: ") or match.group(1) == "Сокращение":
                continue
            for abbr in match.group(1).split(","):
                glossary[abbr.strip()] = match.group(2).strip()
    return glossary


class AbbreviationExpander:
    """Дописывает к запросу полные формы сокращений из глоссария."""

    def __init__(self, glossary: dict[str, str]):
        """
        Args:
            glossary: словарь «сокращение → полная форма».
        """
        self.glossary = glossary
        keys = sorted(glossary, key=len, reverse=True)
        alternatives = "|".join(re.escape(k) for k in keys)
        self.pattern = re.compile(rf"(?<![\w-])({alternatives})(?![\w-])", re.I) if keys else None
        self.lookup = {k.lower(): k for k in keys}

    def find(self, query: str) -> list[str]:
        """Находит сокращения в запросе.

        Args:
            query: текст запроса.

        Returns:
            Сокращения в написании глоссария, без повторов, в порядке появления.
        """
        if self.pattern is None:
            return []
        found = []
        for match in self.pattern.finditer(query):
            key = self.lookup[match.group(1).lower()]
            if key not in found:
                found.append(key)
        return found

    def expand(self, query: str) -> str:
        """Раскрывает сокращения в запросе.

        Args:
            query: текст запроса.

        Returns:
            Запрос с полными формами в скобках, например «ДНП смеси (ДНП — Давление насыщенных паров…)».
        """
        found = self.find(query)
        if not found:
            return query
        expansions = "; ".join(f"{k} — {self.glossary[k]}" for k in found)
        return f"{query} ({expansions})"


@dataclass
class Filters:
    """Ограничения на выдачу.

    Attributes:
        grades: марки бензина; чанк проходит, если упоминает хотя бы одну.
        chunk_types: допустимые типы чанков: text, table.
        doc_ids: допустимые номера документов.
    """

    grades: list[str] | None = None
    chunk_types: list[str] | None = None
    doc_ids: list[int] | None = None

    def mask(self, chunks: list[dict]) -> np.ndarray:
        """Отмечает чанки, прошедшие фильтры.

        Args:
            chunks: чанки корпуса.

        Returns:
            Булев массив длины len(chunks).
        """
        return np.array([self.accepts(c) for c in chunks], dtype=bool)

    def accepts(self, chunk: dict) -> bool:
        """Проверяет один чанк.

        Args:
            chunk: чанк корпуса.

        Returns:
            True, если чанк проходит все заданные фильтры.
        """
        if self.grades and not set(self.grades) & set(chunk.get("grades") or []):
            return False
        if self.chunk_types and chunk.get("chunk_type") not in self.chunk_types:
            return False
        if self.doc_ids and chunk.get("doc_id") not in self.doc_ids:
            return False
        return True


@dataclass
class Hit:
    """Найденный чанк.

    Attributes:
        chunk_id: идентификатор чанка.
        score: оценка метода поиска.
        rank: место в выдаче, начиная с 1.
        chunk: сам чанк.
    """

    chunk_id: str
    score: float
    rank: int
    chunk: dict


def top_k(chunks: list[dict], scores: np.ndarray, k: int, filters: Filters | None = None) -> list[Hit]:
    """Выбирает k лучших чанков по оценкам.

    Args:
        chunks: чанки корпуса.
        scores: оценки всех чанков.
        k: размер выдачи.
        filters: ограничения на выдачу.

    Returns:
        Не более k результатов по убыванию оценки.
    """
    scores = scores.astype(float).copy()
    if filters is not None:
        scores[~filters.mask(chunks)] = -np.inf
    hits = []
    for i in np.argsort(-scores, kind="stable")[:k]:
        if np.isneginf(scores[i]):
            break
        hits.append(Hit(chunks[i]["chunk_id"], float(scores[i]), len(hits) + 1, chunks[i]))
    return hits


class Retriever:
    """Базовый класс поиска: считает оценки всех чанков и возвращает лучшие."""

    name = "base"

    def __init__(self, chunks: list[dict], expander: AbbreviationExpander | None = None):
        """
        Args:
            chunks: чанки корпуса.
            expander: раскрытие сокращений; без него запросы не расширяются.
        """
        self.chunks = chunks
        self.expander = expander

    def scores(self, query: str) -> np.ndarray:
        """Оценивает все чанки.

        Args:
            query: текст запроса.

        Returns:
            Массив оценок длины len(chunks), больше — релевантнее.
        """
        raise NotImplementedError

    def search(self, query: str, k: int = 5, filters: Filters | None = None, expand: bool = False) -> list[Hit]:
        """Ищет чанки по запросу.

        Args:
            query: текст запроса.
            k: размер выдачи.
            filters: ограничения на выдачу.
            expand: раскрывать ли сокращения.

        Returns:
            Не более k результатов по убыванию оценки.
        """
        if expand and self.expander is not None:
            query = self.expander.expand(query)
        return top_k(self.chunks, self.scores(query), k, filters)


class BM25Retriever(Retriever):
    """BM25 по леммам поля text."""

    name = "bm25"

    def __init__(self, chunks: list[dict], expander: AbbreviationExpander | None = None,
                 field: str = "text", k1: float = 1.5, b: float = 0.75):
        """
        Args:
            chunks: чанки корпуса.
            expander: раскрытие сокращений.
            field: поле чанка, по которому строится индекс.
            k1: насыщение частоты термина.
            b: нормализация по длине чанка.
        """
        super().__init__(chunks, expander)
        self.field = field
        self.bm25 = BM25Okapi([tokenize(c[field]) for c in chunks], k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)


class TransformerEncoder:
    """Нейросетевой энкодер sentence-transformers."""

    def __init__(self, model_name: str):
        """
        Args:
            model_name: имя модели на HuggingFace.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.uses_prefixes = "e5" in model_name.lower()

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        """Кодирует тексты в нормированные векторы.

        Args:
            texts: тексты.
            kind: query или passage; для e5 добавляется соответствующий префикс.

        Returns:
            Матрица float32 размера len(texts) × dim.
        """
        if self.uses_prefixes:
            texts = [f"{kind}: {t}" for t in texts]
        vectors = self.model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=None)
def make_encoder(name: str) -> TransformerEncoder:
    """Создаёт энкодер по короткому имени; повторный вызов в том же запуске возвращает уже загруженный.

    Args:
        name: e5, rubert-tiny2 или полное имя модели HuggingFace.

    Returns:
        Энкодер.
    """
    return TransformerEncoder(ENCODERS.get(name, name))


class DenseRetriever(Retriever):
    """Векторный поиск по полю embed_text через FAISS IndexFlatIP."""

    name = "dense"

    def __init__(self, chunks: list[dict], encoder: TransformerEncoder,
                 expander: AbbreviationExpander | None = None, field: str = "embed_text",
                 index: faiss.Index | None = None):
        """
        Args:
            chunks: чанки корпуса.
            encoder: энкодер текстов.
            expander: раскрытие сокращений.
            field: поле чанка для кодирования.
            index: готовый индекс; если не задан, строится заново.
        """
        super().__init__(chunks, expander)
        self.encoder = encoder
        self.field = field
        if index is None:
            vectors = encoder.encode([c.get(field) or c["text"] for c in chunks], "passage")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        self.index = index

    def scores(self, query: str) -> np.ndarray:
        query_vector = self.encoder.encode([query], "query")
        similarities, ids = self.index.search(query_vector, len(self.chunks))
        scores = np.full(len(self.chunks), -1.0)
        scores[ids[0]] = similarities[0]
        return scores

    def save(self, directory: Path, name: str) -> Path:
        """Сохраняет индекс и его описание.

        Args:
            directory: папка для индекса.
            name: имя файлов без расширения.

        Returns:
            Путь к файлу индекса.
        """
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / f"{name}.faiss"
        faiss.write_index(self.index, str(index_path))
        meta = {
            "model": self.encoder.model_name,
            "field": self.field,
            "dim": self.index.d,
            "chunk_ids": [c["chunk_id"] for c in self.chunks],
        }
        meta_path = directory / f"{name}.meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return index_path

    @classmethod
    def load(cls, chunks: list[dict], encoder: TransformerEncoder, directory: Path, name: str,
             expander: AbbreviationExpander | None = None) -> "DenseRetriever":
        """Загружает сохранённый индекс.

        Args:
            chunks: чанки корпуса.
            encoder: энкодер, которым строился индекс.
            directory: папка с индексом.
            name: имя файлов без расширения.
            expander: раскрытие сокращений.

        Returns:
            Векторный поиск с загруженным индексом.

        Raises:
            ValueError: индекс построен по другой версии chunks.jsonl.
        """
        meta = json.loads((directory / f"{name}.meta.json").read_text(encoding="utf-8"))
        if meta["chunk_ids"] != [c["chunk_id"] for c in chunks]:
            raise ValueError("Индекс построен по другой версии chunks.jsonl")
        index = faiss.read_index(str(directory / f"{name}.faiss"))
        return cls(chunks, encoder, expander, field=meta["field"], index=index)


def dense_retriever(chunks: list[dict], encoder_name: str, expander: AbbreviationExpander | None = None,
                    index_dir: Path = INDEX_DIR, rebuild: bool = False) -> DenseRetriever:
    """Загружает сохранённый индекс или строит и сохраняет новый.

    Args:
        chunks: чанки корпуса.
        encoder_name: короткое имя энкодера.
        expander: раскрытие сокращений.
        index_dir: папка с индексами.
        rebuild: построить индекс заново, даже если он сохранён.

    Returns:
        Векторный поиск.
    """
    encoder = make_encoder(encoder_name)
    if not rebuild and (index_dir / f"{encoder_name}.faiss").exists():
        try:
            return DenseRetriever.load(chunks, encoder, index_dir, encoder_name, expander)
        except ValueError:
            pass
    retriever = DenseRetriever(chunks, encoder, expander)
    retriever.save(index_dir, encoder_name)
    return retriever


@lru_cache(maxsize=1)
def default_bm25() -> BM25Retriever:
    """Создаёт BM25 по data/chunks.jsonl один раз за сеанс.

    Returns:
        BM25 с раскрытием сокращений.
    """
    chunks = load_chunks()
    return BM25Retriever(chunks, AbbreviationExpander(load_glossary(chunks)))


def search(query: str, k: int = 5, filters: Filters | None = None, expand: bool = False) -> list[Hit]:
    """Ищет чанки методом BM25.

    Args:
        query: текст запроса.
        k: размер выдачи.
        filters: ограничения на выдачу.
        expand: раскрывать ли сокращения.

    Returns:
        Не более k результатов по убыванию оценки.
    """
    return default_bm25().search(query, k, filters, expand)


def build_retriever(mode: str, encoder_name: str = "e5") -> Retriever:
    """Создаёт поиск нужного вида.

    Args:
        mode: bm25 или dense.
        encoder_name: энкодер для векторного поиска.

    Returns:
        Поиск по data/chunks.jsonl.
    """
    chunks = load_chunks()
    expander = AbbreviationExpander(load_glossary(chunks))
    if mode == "bm25":
        return BM25Retriever(chunks, expander)
    return dense_retriever(chunks, encoder_name, expander)


def print_hits(hits: list[Hit]) -> None:
    """Печатает выдачу: место, чанк, оценку, раздел и начало текста.

    Args:
        hits: результаты поиска.
    """
    for hit in hits:
        chunk = hit.chunk
        path = " > ".join(chunk.get("section_path") or [])
        preview = chunk["text"][:200].replace("\n", " ")
        print(f"{hit.rank}. [{hit.chunk_id}] {hit.score:.4f}  {chunk['doc_title']} > {path}")
        print(f"   {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск по корпусу АССБ")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="построить и сохранить индекс FAISS")
    build.add_argument("--encoder", default="e5")

    find = commands.add_parser("search", help="найти чанки по запросу")
    find.add_argument("query")
    find.add_argument("-k", type=int, default=5)
    find.add_argument("--mode", default="bm25", choices=["bm25", "dense"])
    find.add_argument("--encoder", default="e5")
    find.add_argument("--expand", action="store_true")
    find.add_argument("--grades", nargs="+")
    find.add_argument("--chunk-type", nargs="+", choices=["text", "table"])
    find.add_argument("--doc-id", type=int, nargs="+")
    args = parser.parse_args()

    if args.command == "build":
        retriever = dense_retriever(load_chunks(), args.encoder, rebuild=True)
        print(f"{INDEX_DIR / args.encoder}.faiss: {retriever.index.ntotal} векторов, размерность {retriever.index.d}")
        return

    retriever = build_retriever(args.mode, args.encoder)
    filters = Filters(args.grades, args.chunk_type, args.doc_id)
    if args.expand:
        print("Запрос:", retriever.expander.expand(args.query))
    print_hits(retriever.search(args.query, args.k, filters, args.expand))


if __name__ == "__main__":
    main()
