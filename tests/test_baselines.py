import unittest

from scripts.run_visual_retrieval import hard_negative_metrics, multi_positive_metrics

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
        self.assertEqual(retrieval["map@1"], 0.0)
        self.assertEqual(retrieval["map@2"], 0.5)
        evidence = evidence_metrics([{"a", "b"}], [{"a", "c"}])
        self.assertEqual(evidence["evidence_recall"], 0.5)
        self.assertAlmostEqual(token_f1("甲甲", "甲"), 2 / 3)

    def test_retrieval_average_precision(self):
        metrics = retrieval_metrics(
            [["p1", "x", "p2", "p3"]], [{"p1", "p2", "p3"}], ks=(1, 3, 4)
        )
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["map@1"], 1.0)
        self.assertAlmostEqual(metrics["map@3"], (1.0 + 2 / 3) / 3)
        self.assertAlmostEqual(metrics["map@4"], (1.0 + 2 / 3 + 3 / 4) / 3)

    def test_dynamic_multi_positive_metrics_exclude_single_positive_queries(self):
        metrics = multi_positive_metrics(
            [["wrong", "single"], ["p1", "wrong", "p2"]],
            [{"single"}, {"p1", "p2"}],
        )
        self.assertEqual(metrics["queries"], 1)
        self.assertEqual(metrics["map@R"], 0.5)
        self.assertEqual(metrics["R-Precision"], 0.5)
        self.assertEqual(metrics["R@1"], 1.0)

    def test_dynamic_multi_positive_metrics_use_each_query_R(self):
        metrics = multi_positive_metrics(
            [
                ["p1", "x", "p2", "p3"],
                ["x", "p4", "p5", "p6", "p7"],
            ],
            [{"p1", "p2", "p3"}, {"p4", "p5", "p6", "p7"}],
        )
        self.assertAlmostEqual(metrics["map@R"], ((1.0 + 2 / 3) / 3 + (1 / 2 + 2 / 3 + 3 / 4) / 4) / 2)
        self.assertAlmostEqual(metrics["R-Precision"], (2 / 3 + 3 / 4) / 2)
        self.assertEqual(metrics["R@1"], 0.5)

    def test_hard_negative_ties_are_not_recall_hits(self):
        pairs = [
            {
                "pair_id": "pair-positive",
                "query": "positive text",
                "image_id": "image-positive",
                "image_path": "/images/positive.jpg",
            },
            {
                "pair_id": "pair-negative",
                "query": "negative text",
                "image_id": "image-negative",
                "image_path": "/images/negative.jpg",
            },
        ]
        hard_rows = [
            {
                "description": "positive text",
                "positive_image_path": "/images/positive.jpg",
                "negative_image_paths": ["/images/negative.jpg"],
            },
            {
                "description": "negative text",
                "positive_image_path": "/images/negative.jpg",
                "negative_image_paths": ["/images/positive.jpg"],
            },
        ]
        text_results = [
            {
                "query": "positive text",
                "ranked_image_ids": ["image-positive", "image-negative"],
                "scores": [0.5, 0.5],
            }
        ]
        image_results = [
            {
                "image_id": "image-positive",
                "ranked_pair_ids": ["pair-positive", "pair-negative"],
                "scores": [0.5, 0.5],
            }
        ]

        metrics = hard_negative_metrics(text_results, image_results, hard_rows, pairs)

        self.assertEqual(metrics["T2I_R@1"], 0.0)
        self.assertEqual(metrics["I2T_R@1"], 0.0)
        self.assertEqual(metrics["R@1"], 0.0)
        self.assertEqual(metrics["ties_per_direction"], {"T2I": 1, "I2T": 1})

    def test_r1_requires_positive_at_global_rank_one(self):
        pairs = [
            {"pair_id": "p", "query": "q", "image_id": "i", "image_path": "/i.jpg"},
            {"pair_id": "n", "query": "n", "image_id": "n", "image_path": "/n.jpg"},
            {"pair_id": "x", "query": "x", "image_id": "x", "image_path": "/x.jpg"},
        ]
        hard_rows = [{"description": "q", "positive_image_path": "/i.jpg", "negative_image_paths": ["/n.jpg"]}]
        text_results = [{"query": "q", "ranked_image_ids": ["x", "i", "n"], "scores": [.9, .8, .7]}]
        image_results = []
        metrics = hard_negative_metrics(text_results, image_results, hard_rows, pairs)
        self.assertEqual(metrics["T2I_R@1"], 0.0)
        self.assertEqual(metrics["T2I_HN_Accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
