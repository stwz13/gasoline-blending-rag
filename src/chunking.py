"""Нарезка корпуса АССБ на чанки с таблицами в отдельных чанках и метаданными.

Запуск: python -m src.chunking [--max-chars 1500] [--overlap 200]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
OUTPUT_DIR = ROOT / "data"

MAX_CHARS = 1500
MIN_CHARS = 300
OVERLAP_CHARS = 200
MAX_TABLE_CHARS = 2500

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
MARKER_RE = re.compile(r"^<!--\s*chunk:\s*(\{.*\})\s*-->\s*$")
CAPTION_RE = re.compile(r"^Таблица\s+\d+(\.\d+)*\s+—")
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+(?=[А-ЯЁA-Z0-9«(])")

GRADE_PATTERNS = {
    "АИ-92": [r"АИ-92", r"Регуляр-92"],
    "АИ-95": [r"АИ-95", r"Премиум-95"],
    "АИ-98": [r"АИ-98", r"Супер-98"],
    "АИ-100": [r"АИ-100", r"G-Drive\s*100"],
    "Нормаль-80": [r"Нормаль-80", r"АИ-80"],
    "газовый бензин": [r"газов\w+ бензин"],
}
GRADE_RES = {
    grade: re.compile("|".join(patterns), re.I) for grade, patterns in GRADE_PATTERNS.items()
}


@dataclass
class Chunk:
    """Чанк корпуса с метаданными.

    Attributes:
        chunk_type: text или table.
        block_type: тип блока из разметки: text, recipe, equipment, norm, procedure, component.
        text: исходный фрагмент.
        embed_text: фрагмент с контекстом «документ > раздел» и подписью таблицы для энкодера.
        grades: марки бензина, упомянутые в чанке.
    """

    chunk_id: str
    doc_id: int
    document: str
    doc_title: str
    section: str
    subsection: str
    section_path: list[str]
    chunk_type: str
    block_type: str
    text: str
    embed_text: str
    grades: list[str] = field(default_factory=list)
    category: list[str] = field(default_factory=list)
    method: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    description: str = ""
    table_caption: str = ""
    part: int = 1
    parts_total: int = 1
    topics: list[str] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    position: int = 0
    char_count: int = 0


@dataclass
class Block:
    """Фрагмент раздела: таблица или текст под одной разметкой.

    Attributes:
        kind: text или table.
        marker: метаданные из комментария <!-- chunk -->.
        before: короткий текст перед таблицей.
        after: короткий текст после таблицы.
    """

    kind: str
    lines: list[str]
    marker: dict
    caption: str = ""
    heading_path: tuple[str, ...] = ()
    before: str = ""
    after: str = ""

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def parse_document(raw: str) -> tuple[dict, str]:
    """Отделяет YAML-шапку от тела документа.

    Args:
        raw: текст документа.

    Returns:
        Метаданные шапки и тело документа.
    """
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    return yaml.safe_load(match.group(1)) or {}, raw[match.end():]


def split_blocks(body: str) -> list[Block]:
    """Делит тело документа на блоки по заголовкам, разметке и таблицам.

    Args:
        body: тело документа без YAML-шапки.

    Returns:
        Блоки в порядке следования.
    """
    blocks: list[Block] = []
    path: list[str] = []
    current: Block | None = None
    pending_marker: dict = {}

    def close() -> None:
        nonlocal current
        if current and "".join(current.lines).strip():
            while current.lines and not current.lines[-1].strip():
                current.lines.pop()
            blocks.append(current)
        current = None

    def open_block(kind: str, marker: dict) -> Block:
        return Block(kind=kind, lines=[], marker=marker, heading_path=tuple(path))

    for line in body.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            close()
            pending_marker = {}
            level = len(heading.group(1))
            path[:] = path[: level - 1] + [heading.group(2).strip()]
            continue

        marker = MARKER_RE.match(line)
        if marker:
            close()
            pending_marker = json.loads(marker.group(1))
            continue

        if CAPTION_RE.match(line):
            close()
            current = open_block("table", pending_marker)
            current.caption = line.strip()
            pending_marker = {}
            continue

        if line.startswith("|"):
            if current is None or current.kind != "table":
                close()
                current = open_block("table", pending_marker)
                pending_marker = {}
            current.lines.append(line)
            continue

        if current is not None and current.kind == "table":
            if not line.strip() and not current.lines:
                continue
            close()

        if current is None:
            if not line.strip():
                continue
            current = open_block("text", pending_marker)
            pending_marker = {}
        current.lines.append(line)

    close()
    return blocks


def attach_short_text(blocks: list[Block], min_chars: int = MIN_CHARS) -> list[Block]:
    """Присоединяет короткие неразмеченные тексты к соседнему блоку того же раздела.

    Args:
        blocks: блоки документа.
        min_chars: порог длины короткого текста.

    Returns:
        Блоки без коротких текстовых фрагментов.
    """
    result: list[Block] = []
    for i, block in enumerate(blocks):
        short = block.kind == "text" and not block.marker and len(block.text) < min_chars
        if not short:
            result.append(block)
            continue
        nxt = blocks[i + 1] if i + 1 < len(blocks) else None
        prev = result[-1] if result else None
        if nxt and nxt.heading_path == block.heading_path and nxt.kind == "table":
            nxt.before = f"{nxt.before}\n\n{block.text}".strip()
        elif nxt and nxt.heading_path == block.heading_path and nxt.marker:
            nxt.lines = [*block.lines, "", *nxt.lines]
        elif prev and prev.heading_path == block.heading_path:
            if prev.kind == "table":
                prev.after = f"{prev.after}\n\n{block.text}".strip()
            else:
                prev.lines += ["", *block.lines]
        else:
            result.append(block)
    return result


def paragraphs(text: str) -> list[str]:
    """Делит текст на абзацы, не отрывая вводную фразу и пояснение «где ...» от формулы или списка.

    Args:
        text: текст блока.

    Returns:
        Абзацы.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and (merged[-1].rstrip().endswith(":") or part.startswith("где ")):
            merged[-1] = merged[-1] + "\n\n" + part
        else:
            merged.append(part)
    return merged


def split_long(unit: str, max_chars: int) -> list[str]:
    """Делит длинный абзац по строкам списка, затем по предложениям.

    Args:
        unit: абзац.
        max_chars: максимальная длина части.

    Returns:
        Части абзаца.
    """
    if len(unit) <= max_chars:
        return [unit]
    pieces = unit.split("\n") if "\n" in unit else SENTENCE_RE.split(unit)
    if len(pieces) == 1:
        return [unit[i : i + max_chars] for i in range(0, len(unit), max_chars)]
    out: list[str] = []
    buf = ""
    sep = "\n" if "\n" in unit else " "
    for piece in pieces:
        for sub in split_long(piece, max_chars):
            if buf and len(buf) + len(sep) + len(sub) > max_chars:
                out.append(buf)
                buf = sub
            else:
                buf = f"{buf}{sep}{sub}" if buf else sub
    if buf:
        out.append(buf)
    return out


def tail_overlap(text: str, overlap: int) -> str:
    """Возвращает последние целые предложения фрагмента.

    Args:
        text: фрагмент.
        overlap: максимальная длина перекрытия.

    Returns:
        Перекрытие для начала следующего чанка.
    """
    if overlap <= 0:
        return ""
    sentences = SENTENCE_RE.split(text.replace("\n", " "))
    tail = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {tail}".strip()
        if len(candidate) > overlap:
            break
        tail = candidate
    return tail


def pack_text(text: str, max_chars: int, min_chars: int, overlap: int) -> list[str]:
    """Собирает абзацы в чанки с перекрытием по предложениям.

    Args:
        text: текст блока.
        max_chars: максимальная длина чанка.
        min_chars: короткий последний чанк присоединяется к предыдущему.
        overlap: максимальная длина перекрытия.

    Returns:
        Тексты чанков.
    """
    units = [u for p in paragraphs(text) for u in split_long(p, max_chars)]
    chunks: list[str] = []
    buf = ""
    for unit in units:
        if buf and len(buf) + 2 + len(unit) > max_chars:
            chunks.append(buf)
            prefix = tail_overlap(buf, overlap)
            buf = f"{prefix}\n\n{unit}" if prefix and len(prefix) + 2 + len(unit) <= max_chars else unit
        else:
            buf = f"{buf}\n\n{unit}" if buf else unit
    if buf:
        if chunks and len(buf) < min_chars and len(chunks[-1]) + len(buf) <= max_chars * 1.2:
            chunks[-1] = f"{chunks[-1]}\n\n{buf}"
        else:
            chunks.append(buf)
    return chunks


def split_table(lines: list[str], max_chars: int, row_per_chunk: bool = False) -> list[list[str]]:
    """Делит таблицу по строкам с повтором шапки в каждой части.

    Args:
        lines: строки таблицы Markdown.
        max_chars: максимальная длина части.
        row_per_chunk: по одной строке данных на часть.

    Returns:
        Части таблицы.
    """
    header, rows = lines[:2], lines[2:]
    if row_per_chunk:
        return [header + [row] for row in rows]
    if len("\n".join(lines)) <= max_chars:
        return [lines]
    parts: list[list[str]] = []
    current: list[str] = []
    header_len = len("\n".join(header))
    for row in rows:
        size = header_len + sum(len(r) + 1 for r in current) + len(row)
        if current and size > max_chars:
            parts.append(header + current)
            current = []
        current.append(row)
    if current:
        parts.append(header + current)
    return parts


def as_list(value) -> list[str]:
    """Приводит значение поля разметки к списку строк.

    Args:
        value: строка, список или None.

    Returns:
        Список значений.
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in re.split(r"[;,]", str(value)) if v.strip()]


def detect_grades(*texts: str) -> list[str]:
    """Находит марки бензина с учётом торговых названий.

    Args:
        texts: тексты для поиска.

    Returns:
        Канонические названия марок.
    """
    joined = " ".join(texts)
    return [grade for grade, regex in GRADE_RES.items() if regex.search(joined)]


def context_header(meta: dict, path: tuple[str, ...]) -> str:
    """Формирует строку контекста «документ > раздел > подраздел».

    Args:
        meta: метаданные документа.
        path: путь заголовков блока.

    Returns:
        Строка контекста.
    """
    return " > ".join([meta.get("title", "")] + [p for p in path[1:]])


def chunk_document(
    raw: str,
    max_chars: int = MAX_CHARS,
    min_chars: int = MIN_CHARS,
    overlap: int = OVERLAP_CHARS,
    max_table_chars: int = MAX_TABLE_CHARS,
) -> list[Chunk]:
    """Режет документ на чанки.

    Args:
        raw: текст документа с YAML-шапкой.
        max_chars: максимальная длина текстового чанка.
        min_chars: порог короткого фрагмента.
        overlap: максимальная длина перекрытия.
        max_table_chars: длина, после которой таблица делится по строкам.

    Returns:
        Чанки документа.
    """
    meta, body = parse_document(raw)
    chunks: list[Chunk] = []
    for block in attach_short_text(split_blocks(body), min_chars):
        marker = block.marker
        block_type = marker.get("type", "text")
        text = block.text

        if block.kind == "table":
            is_glossary = "глоссарий" in as_list(marker.get("category"))
            pieces = ["\n".join(p) for p in split_table(block.lines, max_table_chars, is_glossary)]
        else:
            pieces = pack_text(text, max_chars, min_chars, overlap)

        header = context_header(meta, block.heading_path)
        marker_grades = detect_grades(*as_list(marker.get("product")))
        for i, piece in enumerate(pieces, start=1):
            if block.kind == "table":
                before = block.before if i == 1 else ""
                after = block.after if i == len(pieces) else ""
                piece = "\n\n".join(p for p in (before, piece, after) if p)
            embed_parts = [header]
            if block.caption:
                embed_parts.append(block.caption + (f" (часть {i} из {len(pieces)})" if len(pieces) > 1 else ""))
            embed_parts.append(piece)
            grades = sorted(
                set(marker_grades) | set(detect_grades(piece, block.caption)),
                key=list(GRADE_PATTERNS).index,
            )
            chunks.append(
                make_chunk(
                    meta,
                    position=len(chunks),
                    path=block.heading_path,
                    chunk_type=block.kind,
                    block_type=block_type,
                    text=(f"{block.caption}\n\n{piece}" if block.caption else piece),
                    embed_text="\n\n".join(embed_parts),
                    grades=grades,
                    marker=marker,
                    caption=block.caption,
                    part=i,
                    parts_total=len(pieces),
                )
            )
    return chunks


def make_chunk(meta: dict, *, position: int, path: tuple[str, ...], chunk_type: str,
               block_type: str, text: str, embed_text: str, grades: list[str], marker: dict,
               caption: str = "", part: int = 1, parts_total: int = 1) -> Chunk:
    """Создаёт чанк из фрагмента и метаданных документа и разметки.

    Args:
        meta: метаданные документа.
        position: порядковый номер чанка в документе.
        path: путь заголовков.
        marker: метаданные из разметки <!-- chunk -->.

    Returns:
        Чанк.
    """
    doc_id = int(meta.get("doc_id", 0))
    return Chunk(
        chunk_id=f"{doc_id:02d}-{position:03d}",
        doc_id=doc_id,
        document=meta.get("document", ""),
        doc_title=meta.get("title", ""),
        section=path[1] if len(path) > 1 else "",
        subsection=path[2] if len(path) > 2 else "",
        section_path=list(path[1:]),
        chunk_type=chunk_type,
        block_type=block_type,
        text=text,
        embed_text=embed_text,
        grades=grades,
        category=as_list(marker.get("category")),
        method=as_list(marker.get("method")),
        components=as_list(marker.get("components")),
        description=marker.get("description", ""),
        table_caption=caption,
        part=part,
        parts_total=parts_total,
        topics=list(meta.get("topics") or []),
        sources=list(meta.get("sources") or []),
        position=position,
        char_count=len(text),
    )


def chunk_corpus(corpus_dir: Path = CORPUS_DIR, **params) -> list[Chunk]:
    """Режет на чанки все документы корпуса.

    Args:
        corpus_dir: папка с документами.
        params: параметры chunk_document.

    Returns:
        Чанки корпуса.
    """
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("[0-9][0-9]_*.md")):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), **params))
    return chunks


def save_jsonl(chunks: list[Chunk], path: Path) -> None:
    """Сохраняет чанки в JSONL, по одному на строку.

    Args:
        chunks: чанки.
        path: путь к файлу.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def stats(chunks: list[Chunk]) -> dict:
    """Считает статистику нарезки.

    Args:
        chunks: чанки.

    Returns:
        Число чанков по типам и документам, размеры в знаках.
    """
    sizes = sorted(c.char_count for c in chunks)
    by_type = Counter(c.chunk_type for c in chunks)
    return {
        "chunks": len(chunks),
        "by_type": dict(by_type),
        "by_document": dict(Counter(c.document for c in chunks)),
        "chars_min": sizes[0],
        "chars_median": sizes[len(sizes) // 2],
        "chars_mean": round(sum(sizes) / len(sizes)),
        "chars_max": sizes[-1],
        "with_grades": sum(1 for c in chunks if c.grades),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Нарезка корпуса АССБ на чанки")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "chunks.jsonl")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--overlap", type=int, default=OVERLAP_CHARS)
    args = parser.parse_args()

    chunks = chunk_corpus(args.corpus, max_chars=args.max_chars, overlap=args.overlap)
    save_jsonl(chunks, args.out)
    print(f"-> {args.out}")
    print(json.dumps(stats(chunks), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
