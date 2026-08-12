import unittest

from mmmrag.metrics import evidence_metrics, retrieval_metrics, token_f1
from mmmrag.rag import EvidenceAwareRAG
from mmmrag.sparse import SparseRetriever, tokenize


class BaselineTests(unittest.TestCase):
    def test_chinese_tokenization_has_bigrams(self):
        self.assertIn("千佛", tokenize("袈裟千佛纹"))

    def test_sparse_retrieval_prefers_matching_document(self):
        records = [
            {"evidence_id": "a", "content": "第18窟 袈裟 千佛纹"},
            {"evidence_id": "b", "content": "第20窟 面相 高肉髻"},
        ]
        result = SparseRetriever(records).search("千佛纹", top_k=1)
        self.assertEqual(result[0][0]["evidence_id"], "a")

    def test_rag_expands_relation_and_is_sufficient(self):
        corpus = [
            {"evidence_id": "v1", "modality": "visual", "content": "高肉髻", "relations": ["r1"]},
            {"evidence_id": "t1", "modality": "text", "content": "面相丰圆", "relations": ["r1"]},
            {"evidence_id": "r1", "modality": "relation", "content": "主尊 位于 第20窟", "relations": ["v1", "t1"]},
        ]
        sample = {"question_id": "q1", "question": "高肉髻 面相丰圆 主尊位于哪里", "required_modalities": ["visual", "text", "relation"]}
        result = EvidenceAwareRAG(corpus, top_k=1).run(sample)
        self.assertTrue(result["sufficient"])
        self.assertEqual(set(result["evidence_ids"]), {"v1", "t1", "r1"})

    def test_metrics(self):
        retrieval = retrieval_metrics([["a", "b"]], [{"b"}], ks=(1, 2))
        self.assertEqual(retrieval["recall@1"], 0.0)
        self.assertEqual(retrieval["recall@2"], 1.0)
        evidence = evidence_metrics([{"a", "b"}], [{"a", "c"}])
        self.assertEqual(evidence["evidence_recall"], 0.5)
        self.assertAlmostEqual(token_f1("甲甲", "甲"), 2 / 3)


if __name__ == "__main__":
    unittest.main()
