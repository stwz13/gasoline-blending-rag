"""Оценка поиска на размеченных запросах: Recall@5 и MRR@10."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from src.hybrid_retrieval import DEFAULT_ALPHA, RERANKER, HybridRetriever, RerankedRetriever, make_reranker
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
        path: путь к JSONL с полями query_id, tz (номер вопроса из ТЗ, необязательно), category, query, evidence.

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
    """Проверяет, попал ли релевантный чанк в первые k.

    Релевантным считается чанк, содержащий хотя бы одну фразу-доказательство.
    Среднее по запросам даёт долю вопросов с релевантным чанком в топ-k.

    Args:
        found: тексты чанков выдачи по порядку.
        evidence: фразы-доказательства запроса.
        k: глубина выдачи.

    Returns:
        1.0, если релевантный чанк есть среди первых k, иначе 0.0.
    """
    return float(any(contains(text, phrase) for text in found[:k] for phrase in evidence))


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


def trap_rejection(answer_scores: list[float], trap_scores: list[float]) -> float:
    """Считает долю ловушек, которые можно отсечь порогом по оценке лучшего чанка.

    Порог — наименьшая оценка лучшего чанка среди запросов, у которых есть ответ:
    ловушка распознана, если её лучший чанк оценён ниже любого запроса с ответом.

    Args:
        answer_scores: оценки лучшего чанка у запросов с ответом.
        trap_scores: оценки лучшего чанка у ловушек.

    Returns:
        Значение от 0 до 1.
    """
    threshold = min(answer_scores)
    return sum(score < threshold for score in trap_scores) / len(trap_scores)


def evaluate(retriever: Retriever, queries: list[dict], expand: bool = False, k: int = 10) -> dict:
    """Прогоняет запросы через поиск и считает метрики.

    Recall@5 и MRR@10 считаются по запросам с фразами-доказательствами,
    ловушки (запросы без ответа в корпусе) оцениваются отдельно.

    Args:
        retriever: метод поиска.
        queries: размеченные запросы.
        expand: раскрывать ли сокращения.
        k: глубина выдачи для MRR.

    Returns:
        Словарь с ключами recall@5, mrr, by_category, tz, synonyms (вопросы ТЗ С1–С5), traps и per_query.
    """
    per_query = []
    by_category = defaultdict(list)
    trap_scores = []
    for query in queries:
        hits = retriever.search(query["query"], k, expand=expand)
        top_score = hits[0].score if hits else 0.0
        if not query["evidence"]:
            trap_scores.append(top_score)
            continue
        found = [hit.chunk["text"] for hit in hits]
        row = {
            "query_id": query["query_id"],
            "tz": query.get("tz", ""),
            "category": query["category"],
            "recall@5": recall_at_k(found, query["evidence"], 5),
            "rr": reciprocal_rank(found, query["evidence"], k),
            "top_score": top_score,
            "top5": [hit.chunk_id for hit in hits[:5]],
        }
        per_query.append(row)
        by_category[query["category"]].append(row)

    tz_rows = [row for row in per_query if row["tz"]]
    synonym_rows = [row for row in tz_rows if row["tz"].startswith("С")]
    return {
        "recall@5": mean(per_query, "recall@5"),
        "mrr": mean(per_query, "rr"),
        "by_category": {
            category: {"recall@5": mean(rows, "recall@5"), "mrr": mean(rows, "rr")}
            for category, rows in by_category.items()
        },
        "tz": {"recall@5": mean(tz_rows, "recall@5"), "mrr": mean(tz_rows, "rr")} if tz_rows else None,
        "synonyms": (
            {"recall@5": mean(synonym_rows, "recall@5"), "mrr": mean(synonym_rows, "rr")} if synonym_rows else None
        ),
        "traps": trap_rejection([row["top_score"] for row in per_query], trap_scores) if trap_scores else None,
        "per_query": per_query,
    }


def configurations(chunks: list[dict], encoders: list[str], alpha: float,
                   reranker: str | None = RERANKER) -> list[tuple[str, Retriever, bool]]:
    """Перечисляет сравниваемые варианты поиска.

    Args:
        chunks: чанки корпуса.
        encoders: короткие имена энкодеров.
        alpha: вес векторного поиска во взвешенной сумме.
        reranker: кросс-энкодер для переранжирования гибридов; None — без переранжирования.

    Returns:
        Тройки (название, поиск, раскрывать ли сокращения).
    """
    expander = AbbreviationExpander(load_glossary(chunks))
    bm25 = BM25Retriever(chunks, expander)
    cross_encoder = make_reranker(reranker) if reranker and encoders else None
    rerank_cache = {}
    variants = [("BM25", bm25)]
    for encoder in encoders:
        dense = dense_retriever(chunks, encoder, expander)
        rrf = HybridRetriever(bm25, dense, "rrf")
        weighted = HybridRetriever(bm25, dense, "weighted", alpha)
        variants += [
            (f"Вектор ({encoder})", dense),
            (f"Гибрид RRF ({encoder})", rrf),
            (f"Гибрид взвеш. α={alpha} ({encoder})", weighted),
        ]
        if cross_encoder is not None:
            variants += [
                (f"Гибрид RRF ({encoder}) + реранкер", RerankedRetriever(rrf, cross_encoder, cache=rerank_cache)),
                (f"Гибрид взвеш. α={alpha} ({encoder}) + реранкер", RerankedRetriever(weighted, cross_encoder, cache=rerank_cache)),
            ]
    return [
        (f"{name} + сокращения" if expand else name, retriever, expand)
        for name, retriever in variants
        for expand in (False, True)
    ]


def markdown_table(results: dict[str, dict], n_queries: int, n_traps: int) -> str:
    """Собирает таблицу сравнения методов в Markdown.

    Args:
        results: метрики по названиям методов.
        n_queries: число запросов.
        n_traps: число ловушек.

    Returns:
        Таблица; лучшие общие значения выделены жирным.
    """
    best_recall = max(r["recall@5"] for r in results.values())
    best_mrr = max(r["mrr"] for r in results.values())

    def cell(value: float, best: float) -> str:
        return f"**{value:.3f}**" if abs(value - best) < 1e-9 else f"{value:.3f}"

    lines = [
        f"Запросов: {n_queries}, из них ловушек без ответа в корпусе: {n_traps}. "
        "Recall@5 — доля запросов с ответом, у которых релевантный чанк попал в первые 5; MRR@10 — по тем же запросам. "
        "В столбцах категорий, «вопросы ТЗ» и «С1–С5» — Recall@5 / MRR@10. "
        "«Ловушки» — доля ловушек, у которых лучший чанк оценён ниже, чем у любого запроса с ответом.",
        "",
        "| Метод | Recall@5 | MRR@10 | " + " | ".join(CATEGORIES.values()) + " | вопросы ТЗ | С1–С5 | ловушки |",
        "|---|---|---|" + "---|" * (len(CATEGORIES) + 3),
    ]
    for name, r in results.items():
        categories = [
            f"{r['by_category'][c]['recall@5']:.2f} / {r['by_category'][c]['mrr']:.2f}" for c in CATEGORIES
        ]
        tz = f"{r['tz']['recall@5']:.2f} / {r['tz']['mrr']:.2f}" if r["tz"] else "—"
        synonyms = f"{r['synonyms']['recall@5']:.2f} / {r['synonyms']['mrr']:.2f}" if r["synonyms"] else "—"
        traps = f"{r['traps']:.2f}" if r["traps"] is not None else "—"
        lines.append(
            f"| {name} | {cell(r['recall@5'], best_recall)} | {cell(r['mrr'], best_mrr)} | "
            + " | ".join(categories) + f" | {tz} | {synonyms} | {traps} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение BM25, векторного и гибридного поиска")
    parser.add_argument("--encoders", nargs="*", default=["e5"], help="без значений — только BM25")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--reranker", default=RERANKER, help="кросс-энкодер; пустая строка — без переранжирования")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    chunks = load_chunks()
    queries = load_queries(args.queries)
    n_traps = sum(not q["evidence"] for q in queries)
    results = {}
    for name, retriever, expand in configurations(chunks, args.encoders, args.alpha, args.reranker or None):
        results[name] = evaluate(retriever, queries, expand)
        r = results[name]
        traps = f"{r['traps']:.3f}" if r["traps"] is not None else "-"
        print(f"{name:55s} Recall@5={r['recall@5']:.3f}  MRR@10={r['mrr']:.3f}  Traps={traps}")

    report = f"# Сравнение методов поиска\n\n{markdown_table(results, len(queries), n_traps)}\n"
    args.out.write_text(report, encoding="utf-8")
    details_path = args.out.with_suffix(".json")
    details_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{args.out}\n{details_path}")


if __name__ == "__main__":
    main()
