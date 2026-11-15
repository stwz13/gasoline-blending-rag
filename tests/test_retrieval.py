"""Проверки поиска и оценки: python -m unittest discover tests"""

import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from src.chunk_quality import assess, markdown_table
from src.evaluate import (
    CATEGORIES,
    contains,
    evaluate,
    load_queries,
    reciprocal_rank,
    recall_at_k,
    trap_rejection,
)
from src.hybrid_retrieval import HybridRetriever, RerankedRetriever, fuse, minmax, ranks_of
from src.retrieval import (
    AbbreviationExpander,
    BM25Retriever,
    DenseRetriever,
    Filters,
    load_chunks,
    load_glossary,
    search,
    tokenize,
)

CHUNKS = load_chunks()
GLOSSARY = load_glossary(CHUNKS)
QUERIES = load_queries()


class HashingEncoder:
    """Энкодер для тестов: мешок лемм, разложенный по корзинам хеша. Не требует скачивания модели."""

    model_name = "hashing"

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        vectors = np.zeros((len(texts), 256), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, zlib.crc32(token.encode()) % 256] += 1
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


class LengthReranker:
    """Кросс-энкодер для тестов: выше ставит более длинные чанки."""

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.array([len(text) for _, text in pairs], dtype=float)


class TokenizeTest(unittest.TestCase):
    def test_lemmas_and_stopwords(self):
        self.assertEqual(tokenize("Давление в нагнетательных трубопроводах"),
                         ["давление", "нагнетательный", "трубопровод"])

    def test_yo_and_numbers(self):
        self.assertEqual(tokenize("Ещё АИ-95"), ["аи", "95"])


class GlossaryTest(unittest.TestCase):
    def test_glossary_parsed(self):
        self.assertIn("ДНП", GLOSSARY)
        self.assertIn("ГО БКК", GLOSSARY)
        self.assertIn("И100", GLOSSARY)
        self.assertNotIn("Сокращение", GLOSSARY)

    def test_expand_longest_match_and_case(self):
        expander = AbbreviationExpander(GLOSSARY)
        self.assertEqual(expander.find("резервуар РВСПК"), ["РВСПК"])
        self.assertEqual(expander.find("ремонт рвсп"), ["РВСП"])
        self.assertEqual(expander.find("доля го бкк"), ["ГО БКК"])
        self.assertIn("Давление насыщенных паров", expander.expand("ДНП смеси"))
        self.assertEqual(expander.expand("без сокращений"), "без сокращений")


class BM25Test(unittest.TestCase):
    def test_top_hit(self):
        self.assertEqual(search("давление в нагнетательном трубопроводе насоса", k=1)[0].chunk_id, "05-004")

    def test_filters(self):
        hits = search("рецептура", k=10, filters=Filters(grades=["АИ-100"], chunk_types=["table"]))
        self.assertTrue(hits)
        for hit in hits:
            self.assertEqual(hit.chunk["chunk_type"], "table")
            self.assertIn("АИ-100", hit.chunk["grades"])
        hits = search("насос", k=10, filters=Filters(doc_ids=[5]))
        self.assertTrue(all(hit.chunk["doc_id"] == 5 for hit in hits))


class DenseHybridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bm25 = BM25Retriever(CHUNKS)
        cls.dense = DenseRetriever(CHUNKS, HashingEncoder())

    def test_dense_finds_section(self):
        ids = [hit.chunk_id for hit in self.dense.search("отстаивание партии и сброс воды", k=5)]
        self.assertIn("09-006", ids)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as directory:
            self.dense.save(Path(directory), "hashing")
            loaded = DenseRetriever.load(CHUNKS, self.dense.encoder, Path(directory), "hashing")
            query = "калибровка анализатора NIR"
            np.testing.assert_allclose(loaded.scores(query), self.dense.scores(query), rtol=1e-5)

    def test_hybrid(self):
        for method in ("rrf", "weighted"):
            hybrid = HybridRetriever(self.bm25, self.dense, method=method)
            self.assertEqual(len(hybrid.search("ДНП смеси", k=5)), 5)

    def test_rrf_and_minmax(self):
        np.testing.assert_array_equal(ranks_of(np.array([0.1, 0.9, 0.5])), [3, 1, 2])
        np.testing.assert_allclose(minmax(np.array([2.0, 4.0, 3.0])), [0, 1, 0.5])

    def test_rerank(self):
        reranker = RerankedRetriever(self.bm25, LengthReranker(), depth=5)
        base_ids = [hit.chunk_id for hit in self.bm25.search("насос", k=5)]
        hits = reranker.search("насос", k=6)
        self.assertEqual(sorted(hit.chunk_id for hit in hits[:5]), sorted(base_ids))
        lengths = [len(hit.chunk["text"]) for hit in hits[:5]]
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        self.assertNotIn(hits[5].chunk_id, base_ids)

    def test_fuse(self):
        sparse, dense = np.array([3.0, 1.0, 2.0]), np.array([0.1, 0.9, 0.5])
        np.testing.assert_allclose(fuse(sparse, dense, "rrf", rrf_k=0), [1 + 1 / 3, 1 / 3 + 1, 1 / 2 + 1 / 2])
        np.testing.assert_allclose(fuse(sparse, dense, "weighted", alpha=1.0), minmax(dense))
        np.testing.assert_allclose(fuse(sparse, dense, "weighted", alpha=0.0), minmax(sparse))
        with self.assertRaises(ValueError):
            fuse(sparse, dense, "sum")


class EvalTest(unittest.TestCase):
    def test_contains_ignores_case_yo_and_spaces(self):
        self.assertTrue(contains("Партия  отстаивается\nв течение 2 ч.", "партия отстаивается в течение 2 ч"))
        self.assertTrue(contains("Утверждённая рецептура", "утвержденная РЕЦЕПТУРА"))

    def test_metrics(self):
        found = ["первый чанк", "второй чанк про давление", "третий"]
        self.assertEqual(recall_at_k(found, ["давление", "температура"], 5), 1.0)
        self.assertEqual(recall_at_k(found, ["температура"], 5), 0.0)
        self.assertEqual(recall_at_k(found, ["третий"], 2), 0.0)
        self.assertEqual(reciprocal_rank(found, ["давление"]), 0.5)
        self.assertEqual(reciprocal_rank(found, ["температура"]), 0.0)
        self.assertEqual(trap_rejection([0.8, 0.6, 0.9], [0.5, 0.7]), 0.5)

    def test_evaluate_separates_traps(self):
        result = evaluate(BM25Retriever(CHUNKS), QUERIES)
        self.assertEqual(len(result["per_query"]), 73)
        self.assertTrue(0 <= result["traps"] <= 1)
        self.assertTrue(0 <= result["tz"]["recall@5"] <= 1)
        self.assertTrue(0 <= result["synonyms"]["recall@5"] <= 1)


    def test_queries_valid(self):
        self.assertEqual(len(QUERIES), 76)
        self.assertEqual(set(q["category"] for q in QUERIES), set(CATEGORIES) | {"trap"})
        traps = [q for q in QUERIES if q["category"] == "trap"]
        self.assertEqual(len(traps), 3)
        self.assertTrue(all(not q["evidence"] for q in traps))
        tz = [q["tz"] for q in QUERIES if "tz" in q]
        self.assertEqual(len(tz), 25)
        self.assertEqual(len(set(tz)), 25)
        for query in QUERIES:
            if query["category"] != "trap":
                self.assertTrue(query["evidence"], query["query_id"])
            for phrase in query["evidence"]:
                self.assertTrue(any(contains(c["text"], phrase) for c in CHUNKS), phrase)


class ChunkQualityTest(unittest.TestCase):
    def test_assess_all_methods(self):
        results = assess(CHUNKS, QUERIES[:5], {"hashing": HashingEncoder()})
        self.assertEqual(len(results), 4)
        self.assertIn("BM25", results)
        for metrics in results.values():
            self.assertTrue(0 <= metrics["recall@5"] <= 1)
            self.assertTrue(0 <= metrics["mrr"] <= 1)

    def test_table_marks_best(self):
        results = {
            (500, 100): {"BM25": {"recall@5": 0.8, "mrr": 0.7}},
            (1500, 200): {"BM25": {"recall@5": 0.9, "mrr": 0.6}},
        }
        table = markdown_table(results, 60)
        self.assertIn("| 500 / 100 | 0.800 / **0.700** |", table)
        self.assertIn("| **1500 / 200** (текущий) | **0.900** / 0.600 |", table)


if __name__ == "__main__":
    unittest.main()
