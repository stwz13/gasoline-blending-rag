"""Сравнение вариантов нарезки корпуса без нейросетевых моделей.

Для каждого варианта считаются свойства самих чанков и качество поиска BM25
по фразам-доказательствам из data/queries.jsonl.
"""

import argparse
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median

from src.chunking import CORPUS_DIR, MAX_CHARS, OVERLAP_CHARS, SENTENCE_RE, chunk_corpus, parse_document
from src.evaluate import DATA_DIR, contains, evaluate, load_queries, normalize
from src.retrieval import BM25Retriever

RESULTS_PATH = DATA_DIR / "chunk_quality.md"

VARIANTS = [
    (500, 100),
    (800, 150),
    (1500, 0),
    (MAX_CHARS, OVERLAP_CHARS),
    (2500, 300),
]

SENTENCE_END_RE = re.compile(r"[.!?:;…»)]$")
SERVICE_LINE_RE = re.compile(r"^(#|<!--|\|[-:| ]+\|$|\$\$)")


def corpus_units(corpus_dir: Path) -> list[str]:
    """Делит корпус на проверочные единицы: предложения и строки таблиц.

    Args:
        corpus_dir: папка с документами.

    Returns:
        Единицы длиной от 20 знаков без заголовков, разметки и разделителей таблиц.
    """
    units = []
    for path in sorted(corpus_dir.glob("[0-9][0-9]_*.md")):
        _, body = parse_document(path.read_text(encoding="utf-8"))
        for line in body.splitlines():
            line = line.strip()
            if not line or SERVICE_LINE_RE.match(line):
                continue
            parts = [line] if line.startswith("|") else SENTENCE_RE.split(line)
            units.extend(part for part in parts if len(part) >= 20)
    return units


def cut_mid_sentence(chunks: list[dict]) -> float:
    """Считает долю текстовых чанков, оборванных посреди предложения.

    Args:
        chunks: чанки варианта нарезки.

    Returns:
        Доля от 0 до 1.
    """
    texts = [c["text"].rstrip() for c in chunks if c["chunk_type"] == "text"]
    return sum(not SENTENCE_END_RE.search(t) for t in texts) / len(texts)


def table_split(chunks: list[dict]) -> tuple[int, float]:
    """Проверяет разрезанные таблицы.

    Args:
        chunks: чанки варианта нарезки.

    Returns:
        Число таблиц, разрезанных на части, и доля таких таблиц, у которых каждая часть повторяет шапку.
    """
    parts = defaultdict(list)
    for chunk in chunks:
        if chunk["chunk_type"] == "table" and chunk["parts_total"] > 1:
            parts[(chunk["doc_id"], chunk["table_caption"])].append(chunk["text"])
    if not parts:
        return 0, 1.0

    def header(text: str) -> str:
        return next((line for line in text.splitlines() if line.startswith("|")), "")

    with_header = sum(len({header(t) for t in texts}) == 1 for texts in parts.values())
    return len(parts), with_header / len(parts)


def coverage(chunks: list[dict], units: list[str]) -> float:
    """Считает долю единиц корпуса, целиком вошедших хотя бы в один чанк.

    Args:
        chunks: чанки варианта нарезки.
        units: предложения и строки таблиц корпуса.

    Returns:
        Доля от 0 до 1.
    """
    joined = "\n".join(normalize(c["text"]) for c in chunks)
    return sum(normalize(unit) in joined for unit in units) / len(units)


def evidence_kept(chunks: list[dict], queries: list[dict]) -> float:
    """Считает долю фраз-доказательств, целиком попавших хотя бы в один чанк.

    Args:
        chunks: чанки варианта нарезки.
        queries: размеченные запросы.

    Returns:
        Доля от 0 до 1; меньше 1, если нарезка разрезала ответ между чанками.
    """
    phrases = [phrase for query in queries for phrase in query["evidence"]]
    return sum(any(contains(c["text"], phrase) for c in chunks) for phrase in phrases) / len(phrases)


def assess(chunks: list[dict], units: list[str], queries: list[dict], corpus_chars: int) -> dict:
    """Считает все показатели одного варианта нарезки.

    Args:
        chunks: чанки варианта нарезки.
        units: предложения и строки таблиц корпуса.
        queries: размеченные запросы.
        corpus_chars: длина корпуса в знаках без служебных строк.

    Returns:
        Словарь показателей.
    """
    sizes = [len(c["text"]) for c in chunks if c["chunk_type"] == "text"]
    tables_split, headers_kept = table_split(chunks)
    search = evaluate(BM25Retriever(chunks), queries)
    return {
        "chunks": len(chunks),
        "median": median(sizes),
        "max": max(sizes),
        "cut": cut_mid_sentence(chunks),
        "tables_split": tables_split,
        "headers_kept": headers_kept,
        "coverage": coverage(chunks, units),
        "volume": sum(len(c["text"]) for c in chunks) / corpus_chars,
        "evidence_kept": evidence_kept(chunks, queries),
        "recall@5": search["recall@5"],
        "mrr": search["mrr"],
    }


def markdown_table(results: dict[tuple[int, int], dict]) -> str:
    """Собирает таблицу сравнения вариантов нарезки.

    Args:
        results: показатели по парам (max_chars, overlap).

    Returns:
        Таблица в Markdown с пояснениями к столбцам.
    """
    lines = [
        "| max_chars / overlap | Чанков | Медиана, знаков | Максимум | Оборваны | Таблиц разрезано | "
        "Шапка повторена | Покрытие | Объём / корпус | Ответ в одном чанке | BM25 Recall@5 | BM25 MRR@10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (max_chars, overlap), r in results.items():
        name = f"{max_chars} / {overlap}"
        if (max_chars, overlap) == (MAX_CHARS, OVERLAP_CHARS):
            name = f"**{name}** (текущий)"
        lines.append(
            f"| {name} | {r['chunks']} | {r['median']:.0f} | {r['max']} | {r['cut']:.0%} | {r['tables_split']} | "
            f"{r['headers_kept']:.0%} | {r['coverage']:.1%} | {r['volume']:.2f} | {r['evidence_kept']:.0%} | "
            f"{r['recall@5']:.3f} | {r['mrr']:.3f} |"
        )
    lines += [
        "",
        "- **Медиана, Максимум** — длина текстовых чанков.",
        "- **Оборваны** — доля текстовых чанков, которые не заканчиваются знаком конца предложения.",
        "- **Шапка повторена** — доля разрезанных таблиц, у которых каждая часть начинается с той же шапки.",
        "- **Покрытие** — доля предложений и строк таблиц корпуса, целиком вошедших хотя бы в один чанк.",
        "- **Объём / корпус** — суммарная длина чанков к длине корпуса; больше 1 за счёт перекрытия и подписей таблиц.",
        "- **Ответ в одном чанке** — доля фраз-доказательств из запросов, которые не разрезаны между чанками.",
        "- **BM25 Recall@5, MRR@10** — качество поиска по тем же фразам-доказательствам.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение вариантов нарезки корпуса")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    units = corpus_units(args.corpus)
    corpus_chars = sum(len(unit) for unit in units)
    queries = load_queries()
    results = {}
    for max_chars, overlap in VARIANTS:
        chunks = [asdict(c) for c in chunk_corpus(args.corpus, max_chars=max_chars, overlap=overlap)]
        results[(max_chars, overlap)] = assess(chunks, units, queries, corpus_chars)
        r = results[(max_chars, overlap)]
        print(f"{max_chars:>5} / {overlap:<4} чанков={r['chunks']:<4} ответ целиком={r['evidence_kept']:.0%}  "
              f"Recall@5={r['recall@5']:.3f}  MRR@10={r['mrr']:.3f}")

    args.out.write_text(f"# Сравнение вариантов нарезки\n\n{markdown_table(results)}\n", encoding="utf-8")
    print(f"\n{args.out}")


if __name__ == "__main__":
    main()
