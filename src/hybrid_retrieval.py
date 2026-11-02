"""Гибридный поиск: объединение BM25 и векторного поиска."""

import argparse

import numpy as np

from src.retrieval import (
    AbbreviationExpander,
    BM25Retriever,
    Filters,
    Retriever,
    dense_retriever,
    load_chunks,
    load_glossary,
    print_hits,
)

RRF_K = 60
DEFAULT_ALPHA = 0.5


def minmax(values: np.ndarray) -> np.ndarray:
    """Приводит значения к отрезку [0, 1].

    Args:
        values: исходные значения.

    Returns:
        Нормированные значения; нули, если все значения равны.
    """
    low, high = values.min(), values.max()
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def ranks_of(scores: np.ndarray) -> np.ndarray:
    """Переводит оценки в места.

    Args:
        scores: оценки чанков.

    Returns:
        Место каждого чанка, 1 — лучший.
    """
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def fuse(sparse_scores: np.ndarray, dense_scores: np.ndarray, method: str,
         alpha: float = DEFAULT_ALPHA, rrf_k: int = RRF_K) -> np.ndarray:
    """Объединяет оценки BM25 и векторного поиска.

    Args:
        sparse_scores: оценки BM25.
        dense_scores: оценки векторного поиска.
        method: rrf или weighted.
        alpha: вес векторного поиска во взвешенной сумме.
        rrf_k: сглаживающая константа RRF.

    Returns:
        Итоговые оценки чанков.

    Raises:
        ValueError: неизвестный метод.
    """
    if method == "rrf":
        return 1 / (rrf_k + ranks_of(sparse_scores)) + 1 / (rrf_k + ranks_of(dense_scores))
    if method == "weighted":
        return alpha * minmax(dense_scores) + (1 - alpha) * minmax(sparse_scores)
    raise ValueError(f"Неизвестный метод гибрида: {method}")


class HybridRetriever(Retriever):
    """Объединение BM25 и векторного поиска.

    rrf: 1 / (k + место в BM25) + 1 / (k + место в векторном поиске).
    weighted: alpha · minmax(вектор) + (1 − alpha) · minmax(BM25).
    """

    def __init__(self, sparse: Retriever, dense: Retriever, method: str = "rrf",
                 alpha: float = DEFAULT_ALPHA, rrf_k: int = RRF_K):
        """
        Args:
            sparse: поиск по словам.
            dense: векторный поиск.
            method: rrf или weighted.
            alpha: вес векторного поиска во взвешенной сумме.
            rrf_k: сглаживающая константа RRF.

        Raises:
            ValueError: неизвестный метод.
        """
        if method not in ("rrf", "weighted"):
            raise ValueError(f"Неизвестный метод гибрида: {method}")
        super().__init__(sparse.chunks, sparse.expander)
        self.sparse = sparse
        self.dense = dense
        self.method = method
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.name = f"hybrid-{method}"

    def scores(self, query: str) -> np.ndarray:
        return fuse(self.sparse.scores(query), self.dense.scores(query), self.method, self.alpha, self.rrf_k)


def build_hybrid(method: str = "rrf", encoder_name: str = "e5", alpha: float = DEFAULT_ALPHA,
                 rrf_k: int = RRF_K) -> HybridRetriever:
    """Создаёт гибридный поиск по data/chunks.jsonl.

    Args:
        method: rrf или weighted.
        encoder_name: энкодер для векторного поиска.
        alpha: вес векторного поиска во взвешенной сумме.
        rrf_k: сглаживающая константа RRF.

    Returns:
        Гибридный поиск с раскрытием сокращений.
    """
    chunks = load_chunks()
    expander = AbbreviationExpander(load_glossary(chunks))
    sparse = BM25Retriever(chunks, expander)
    dense = dense_retriever(chunks, encoder_name, expander)
    return HybridRetriever(sparse, dense, method, alpha, rrf_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Гибридный поиск по корпусу АССБ")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--method", default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--encoder", default="e5")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--grades", nargs="+")
    parser.add_argument("--chunk-type", nargs="+", choices=["text", "table"])
    parser.add_argument("--doc-id", type=int, nargs="+")
    args = parser.parse_args()

    retriever = build_hybrid(args.method, args.encoder, args.alpha, args.rrf_k)
    filters = Filters(args.grades, args.chunk_type, args.doc_id)
    if args.expand:
        print("Запрос:", retriever.expander.expand(args.query))
    print_hits(retriever.search(args.query, args.k, filters, args.expand))


if __name__ == "__main__":
    main()
