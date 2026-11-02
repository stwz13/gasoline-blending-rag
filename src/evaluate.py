"""Оценка поиска на размеченных запросах: Recall@5 и MRR@10."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src.hybrid_retrieval import DEFAULT_ALPHA, HybridRetriever
from src.retrieval import (
    DATA_DIR,
    AbbreviationExpander,
    BM25Retriever,
    Retriever,
    dense_retriever,
    load_chunks,
    load_glossary,
)

QUERIES_PATH = DATA_DIR / "queries.jsonl"
RESULTS_PATH = DATA_DIR / "eval_results.md"

CATEGORIES = {
    "factoid": "факт/число",
    "procedure": "процедура",
    "table": "таблица",
    "abbreviation": "сокращения",
    "paraphrase": "перефраз",
}


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Читает размеченные запросы.

    Args:
        path: путь к JSONL с полями query_id, category, query, evidence.

    Returns:
        Список запросов.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str) -> str:
    """Приводит текст к виду для сравнения строк.

    Args:
        text: исходный текст.

    Returns:
        Текст в нижнем регистре, «ё» заменена на «е», пробелы схлопнуты.
    """
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def contains(chunk_text: str, phrase: str) -> bool:
    """Проверяет, что фраза-доказательство целиком входит в чанк.

    Args:
        chunk_text: текст чанка.
        phrase: фраза из разметки запроса.

    Returns:
        True, если фраза найдена.
    """
    return normalize(phrase) in normalize(chunk_text)


def recall_at_k(found: list[str], evidence: list[str], k: int = 5) -> float:
    """Считает долю фраз-доказательств, попавших в первые k чанков.

    Args:
        found: тексты чанков выдачи по порядку.
        evidence: фразы-доказательства запроса.
        k: глубина выдачи.

    Returns:
        Значение от 0 до 1.
    """
    covered = sum(any(contains(text, phrase) for text in found[:k]) for phrase in evidence)
    return covered / len(evidence)


def reciprocal_rank(found: list[str], evidence: list[str], k: int = 10) -> float:
    """Считает обратное место первого чанка с доказательством.

    Args:
        found: тексты чанков выдачи по порядку.
        evidence: фразы-доказательства запроса.
        k: глубина выдачи.

    Returns:
        1 / место первого чанка, содержащего хотя бы одну фразу, или 0, если его нет среди первых k.
    """
    for rank, text in enumerate(found[:k], start=1):
        if any(contains(text, phrase) for phrase in evidence):
            return 1 / rank
    return 0.0


def mean(rows: list[dict], key: str) -> float:
    """Считает среднее значение поля.

    Args:
        rows: результаты по запросам.
        key: имя поля.

    Returns:
        Среднее значение.
    """
    return sum(row[key] for row in rows) / len(rows)


def evaluate(retriever: Retriever, queries: list[dict], expand: bool = False, k: int = 10) -> dict:
    """Прогоняет запросы через поиск и считает метрики.

    Args:
        retriever: метод поиска.
        queries: размеченные запросы.
        expand: раскрывать ли сокращения.
        k: глубина выдачи для MRR.

    Returns:
        Словарь с ключами recall@5, mrr, by_category и per_query.
    """
    per_query = []
    by_category = defaultdict(list)
    for query in queries:
        hits = retriever.search(query["query"], k, expand=expand)
        found = [hit.chunk["text"] for hit in hits]
        row = {
            "query_id": query["query_id"],
            "category": query["category"],
            "recall@5": recall_at_k(found, query["evidence"], 5),
            "rr": reciprocal_rank(found, query["evidence"], k),
            "top5": [hit.chunk_id for hit in hits[:5]],
        }
        per_query.append(row)
        by_category[query["category"]].append(row)

    return {
        "recall@5": mean(per_query, "recall@5"),
        "mrr": mean(per_query, "rr"),
        "by_category": {
            category: {"recall@5": mean(rows, "recall@5"), "mrr": mean(rows, "rr")}
            for category, rows in by_category.items()
        },
        "per_query": per_query,
    }


def configurations(chunks: list[dict], encoders: list[str], alpha: float) -> list[tuple[str, Retriever, bool]]:
    """Перечисляет сравниваемые варианты поиска.

    Args:
        chunks: чанки корпуса.
        encoders: короткие имена энкодеров.
        alpha: вес векторного поиска во взвешенной сумме.

    Returns:
        Тройки (название, поиск, раскрывать ли сокращения).
    """
    expander = AbbreviationExpander(load_glossary(chunks))
    bm25 = BM25Retriever(chunks, expander)
    variants = [("BM25", bm25)]
    for encoder in encoders:
        dense = dense_retriever(chunks, encoder, expander)
        variants += [
            (f"Вектор ({encoder})", dense),
            (f"Гибрид RRF ({encoder})", HybridRetriever(bm25, dense, "rrf")),
            (f"Гибрид взвеш. α={alpha} ({encoder})", HybridRetriever(bm25, dense, "weighted", alpha)),
        ]
    return [
        (f"{name} + сокращения" if expand else name, retriever, expand)
        for name, retriever in variants
        for expand in (False, True)
    ]


def markdown_table(results: dict[str, dict], n_queries: int) -> str:
    """Собирает таблицу сравнения методов в Markdown.

    Args:
        results: метрики по названиям методов.
        n_queries: число запросов.

    Returns:
        Таблица; лучшие общие значения выделены жирным.
    """
    best_recall = max(r["recall@5"] for r in results.values())
    best_mrr = max(r["mrr"] for r in results.values())

    def cell(value: float, best: float) -> str:
        return f"**{value:.3f}**" if abs(value - best) < 1e-9 else f"{value:.3f}"

    lines = [
        f"Запросов: {n_queries}. В столбцах категорий — Recall@5 / MRR@10.",
        "",
        "| Метод | Recall@5 | MRR@10 | " + " | ".join(CATEGORIES.values()) + " |",
        "|---|---|---|" + "---|" * len(CATEGORIES),
    ]
    for name, r in results.items():
        categories = [
            f"{r['by_category'][c]['recall@5']:.2f} / {r['by_category'][c]['mrr']:.2f}" for c in CATEGORIES
        ]
        lines.append(
            f"| {name} | {cell(r['recall@5'], best_recall)} | {cell(r['mrr'], best_mrr)} | "
            + " | ".join(categories) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение BM25, векторного и гибридного поиска")
    parser.add_argument("--encoders", nargs="*", default=["e5"], help="без значений — только BM25")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    chunks = load_chunks()
    queries = load_queries(args.queries)
    results = {}
    for name, retriever, expand in configurations(chunks, args.encoders, args.alpha):
        results[name] = evaluate(retriever, queries, expand)
        print(f"{name:45s} Recall@5={results[name]['recall@5']:.3f}  MRR@10={results[name]['mrr']:.3f}")

    args.out.write_text(f"# Сравнение методов поиска\n\n{markdown_table(results, len(queries))}\n", encoding="utf-8")
    details_path = args.out.with_suffix(".json")
    details_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{args.out}\n{details_path}")


if __name__ == "__main__":
    main()
