"""Сравнение вариантов нарезки корпуса по качеству поиска.

Для каждого варианта нарезки строятся BM25, векторный и гибридный поиск
и считаются Recall@5 и MRR@10 по фразам-доказательствам из data/queries.jsonl.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from src.chunking import CORPUS_DIR, MAX_CHARS, OVERLAP_CHARS, chunk_corpus
from src.evaluate import DATA_DIR, evaluate, load_queries
from src.hybrid_retrieval import DEFAULT_ALPHA, HybridRetriever
from src.retrieval import BM25Retriever, DenseRetriever, Retriever, TransformerEncoder, make_encoder

RESULTS_PATH = DATA_DIR / "chunk_quality.md"

VARIANTS = [
    (500, 100),
    (800, 150),
    (1500, 0),
    (MAX_CHARS, OVERLAP_CHARS),
    (1500, 300),
    (2500, 300),
]


def retrievers(chunks: list[dict], encoders: dict[str, TransformerEncoder]) -> dict[str, Retriever]:
    """Создаёт методы поиска для одного варианта нарезки.

    Args:
        chunks: чанки варианта нарезки.
        encoders: энкодеры по коротким именам; без них оценивается только BM25.

    Returns:
        Методы поиска по названиям. Векторный индекс строится заново и не сохраняется.
    """
    bm25 = BM25Retriever(chunks)
    methods = {"BM25": bm25}
    for name, encoder in encoders.items():
        dense = DenseRetriever(chunks, encoder)
        methods[f"Вектор ({name})"] = dense
        methods[f"Гибрид RRF ({name})"] = HybridRetriever(bm25, dense, "rrf")
        methods[f"Гибрид взвеш. α={DEFAULT_ALPHA} ({name})"] = HybridRetriever(bm25, dense, "weighted")
    return methods


def assess(chunks: list[dict], queries: list[dict], encoders: dict[str, TransformerEncoder]) -> dict[str, dict]:
    """Считает качество поиска на одном варианте нарезки.

    Args:
        chunks: чанки варианта нарезки.
        queries: размеченные запросы.
        encoders: энкодеры для векторного и гибридного поиска.

    Returns:
        Recall@5 и MRR@10 по названиям методов поиска.
    """
    results = {}
    for name, retriever in retrievers(chunks, encoders).items():
        result = evaluate(retriever, queries)
        results[name] = {"recall@5": result["recall@5"], "mrr": result["mrr"]}
    return results


def variant_name(max_chars: int, overlap: int) -> str:
    """Подписывает вариант нарезки.

    Args:
        max_chars: максимальная длина текстового чанка.
        overlap: длина перекрытия.

    Returns:
        Подпись «max_chars / overlap»; текущий вариант выделен.
    """
    name = f"{max_chars} / {overlap}"
    return f"**{name}** (текущий)" if (max_chars, overlap) == (MAX_CHARS, OVERLAP_CHARS) else name


def markdown_table(results: dict[tuple[int, int], dict[str, dict]], n_queries: int) -> str:
    """Собирает таблицу «вариант нарезки × метод поиска».

    Args:
        results: метрики по парам (max_chars, overlap) и методам поиска.
        n_queries: число запросов с ответом в корпусе.

    Returns:
        Таблица в Markdown; лучшее значение каждого метода выделено жирным.
    """
    methods = list(next(iter(results.values())))
    best = {
        method: (max(r[method]["recall@5"] for r in results.values()), max(r[method]["mrr"] for r in results.values()))
        for method in methods
    }

    def cell(value: float, top: float) -> str:
        return f"**{value:.3f}**" if abs(value - top) < 1e-9 else f"{value:.3f}"

    lines = [
        f"Запросов с ответом: {n_queries}. В ячейках Recall@5 / MRR@10; жирным — лучшее значение метода среди вариантов нарезки.",
        "",
        "| max_chars / overlap | " + " | ".join(methods) + " |",
        "|---|" + "---|" * len(methods),
    ]
    for (max_chars, overlap), r in results.items():
        cells = [f"{cell(r[m]['recall@5'], best[m][0])} / {cell(r[m]['mrr'], best[m][1])}" for m in methods]
        lines.append(f"| {variant_name(max_chars, overlap)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение вариантов нарезки по качеству поиска")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--encoders", nargs="*", default=["e5"], help="без значений — только BM25")
    args = parser.parse_args()

    queries = load_queries()
    encoders = {name: make_encoder(name) for name in args.encoders}
    results = {}
    for max_chars, overlap in VARIANTS:
        chunks = [asdict(c) for c in chunk_corpus(args.corpus, max_chars=max_chars, overlap=overlap)]
        results[(max_chars, overlap)] = assess(chunks, queries, encoders)
        print(f"{max_chars:>5} / {overlap:<4} чанков={len(chunks)}")
        for name, metrics in results[(max_chars, overlap)].items():
            print(f"    {name:35s} Recall@5={metrics['recall@5']:.3f}  MRR@10={metrics['mrr']:.3f}")

    report = markdown_table(results, sum(bool(q["evidence"]) for q in queries))
    args.out.write_text(f"# Сравнение вариантов нарезки\n\n{report}\n", encoding="utf-8")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
