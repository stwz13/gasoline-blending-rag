from src.generate import INSTRUCTIONS, NOT_FOUND, build_messages, source_label
from src.retrieval import Hit


def make_hit(chunk_id: str, text: str) -> Hit:
    chunk = {"chunk_id": chunk_id, "doc_title": "Рецептуры", "section": "3. АИ-95", "subsection": "", "text": text}
    return Hit(chunk_id, 1.0, 1, chunk)


def test_messages_number_fragments_and_keep_question():
    hits = [make_hit("03-001", "Риформат 40 %"), make_hit("03-002", "Алкилат 20 %")]
    [user] = build_messages("Какая доля риформата?", hits)
    assert user["role"] == "user" and user["content"].startswith(INSTRUCTIONS)
    assert user["content"].index("[1] ") < user["content"].index("[2] ")
    assert "Риформат 40 %" in user["content"] and user["content"].endswith("Вопрос: Какая доля риформата?")


def test_source_label_names_document_section_and_chunk():
    assert source_label(make_hit("03-001", "x")) == "Рецептуры, 3. АИ-95 (чанк 03-001)"


def test_prompt_mentions_paraphrase_and_refusal():
    assert "другими словами" in INSTRUCTIONS and NOT_FOUND in INSTRUCTIONS
