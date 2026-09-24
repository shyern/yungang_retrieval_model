from __future__ import annotations

from collections import defaultdict

from .sparse import SparseRetriever


class EvidenceAwareRAG:
    def __init__(self, corpus: list[dict], top_k: int = 1, max_iterations: int = 2):
        self.corpus = corpus
        self.top_k = top_k
        self.max_iterations = max_iterations
        grouped = defaultdict(list)
        for record in corpus:
            grouped[record["modality"]].append(record)
        self.retrievers = {modality: SparseRetriever(records) for modality, records in grouped.items()}
        self.by_id = {record["evidence_id"]: record for record in corpus}

    def run(self, sample: dict) -> dict:
        required = sample.get("required_modalities", ["visual", "text"])
        selected: dict[str, tuple[dict, float, int]] = {}
        query = sample["question"]
        trace = []

        for iteration in range(self.max_iterations + 1):
            missing = self._missing_modalities(selected, required)
            targets = missing or required
            round_items = []
            for modality in targets:
                retriever = self.retrievers.get(modality)
                if not retriever:
                    continue
                hint = sample.get("target_hints", {}).get(modality, "")
                targeted_query = f"{query} {hint}" if iteration else query
                for record, score in retriever.search(targeted_query, self.top_k):
                    evidence_id = record["evidence_id"]
                    previous = selected.get(evidence_id)
                    if previous is None or score > previous[1]:
                        selected[evidence_id] = (record, score, iteration)
                    round_items.append(evidence_id)

            self._expand_relations(selected, iteration)
            missing_after = self._missing_modalities(selected, required)
            trace.append({"iteration": iteration, "target_modalities": targets, "retrieved_ids": round_items, "missing_after": missing_after})
            if not missing_after:
                break
            query = f"{sample['question']} {' '.join(sample.get('target_hints', {}).get(item, item) for item in missing_after)}"

        ranked = sorted(selected.values(), key=lambda item: (item[2], -item[1], item[0]["evidence_id"]))
        evidence = [item[0] for item in ranked]
        return {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "prediction": self._extractive_answer(evidence),
            "evidence_ids": [item["evidence_id"] for item in evidence],
            "evidence": evidence,
            "sufficient": not self._missing_modalities(selected, required),
            "trace": trace,
        }

    def _expand_relations(self, selected: dict, iteration: int) -> None:
        related_ids = {related for record, _, _ in list(selected.values()) for related in record.get("relations", [])}
        for related_id in related_ids:
            if related_id in self.by_id and related_id not in selected:
                selected[related_id] = (self.by_id[related_id], 0.0, iteration)

    @staticmethod
    def _missing_modalities(selected: dict, required: list[str]) -> list[str]:
        present = {record["modality"] for record, _, _ in selected.values()}
        return [modality for modality in required if modality not in present]

    @staticmethod
    def _extractive_answer(evidence: list[dict]) -> str:
        text = next((item["content"] for item in evidence if item["modality"] == "text"), "")
        relation = next((item["content"] for item in evidence if item["modality"] == "relation"), "")
        return " ".join(part for part in (text, relation) if part)
