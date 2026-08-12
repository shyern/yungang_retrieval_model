from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character bigrams without dependencies."""
    text = text.lower()
    latin = re.findall(r"[a-z0-9_]+", text)
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    chinese = []
    for chunk in chinese_chunks:
        chinese.extend(list(chunk))
        chinese.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return latin + chinese


class SparseRetriever:
    def __init__(self, records: list[dict]):
        self.records = records
        self.documents = [tokenize(record.get("content", "")) for record in records]
        self.document_frequency = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))
        self.average_length = sum(map(len, self.documents)) / max(len(self.documents), 1)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        query_tokens = tokenize(query)
        scored = [(record, self._score(query_tokens, document)) for record, document in zip(self.records, self.documents)]
        scored.sort(key=lambda item: (-item[1], item[0].get("evidence_id", "")))
        return scored[:top_k]

    def _score(self, query: list[str], document: list[str]) -> float:
        if not document:
            return 0.0
        frequencies = Counter(document)
        score = 0.0
        k1, b = 1.5, 0.75
        for token in query:
            df = self.document_frequency.get(token, 0)
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            tf = frequencies[token]
            denominator = tf + k1 * (1 - b + b * len(document) / max(self.average_length, 1))
            score += idf * tf * (k1 + 1) / denominator if denominator else 0.0
        return score

