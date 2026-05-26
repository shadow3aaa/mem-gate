import json
import math
import re
from collections import Counter
from typing import Any

from eval import MemGateEvaluator
from foo_system_ollama import FooSystem
from hybrid_bert_gate_system import HybridBertGateSystem
from hybrid_broad_recall_system import HybridBroadRecallSystem


class MySystem:
    def __init__(self):
        self.memories: list[dict[str, Any]] = []
        self.memory_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_tf: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0
        self.built = False

    def index(self, memory_id: str, memory: dict[str, Any]) -> None:
        text = self._memory_text(memory)
        tokens = self._tokenize(text)

        self.memories.append(memory)
        self.memory_ids.append(memory_id)
        self.doc_tokens.append(tokens)
        self.doc_tf.append(Counter(tokens))

        for token in set(tokens):
            self.df[token] += 1

        self.built = False

    def recall(self, user_input: str, top_k: int) -> list[str]:
        self._build()

        if not self.memory_ids:
            return []

        query_terms = self._tokenize(user_input)
        if not query_terms:
            return self.memory_ids[:top_k]

        scores: list[tuple[float, str]] = []
        n_docs = len(self.memory_ids)

        for memory_id, tokens, tf in zip(self.memory_ids, self.doc_tokens, self.doc_tf):
            dl = len(tokens)
            score = 0.0

            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue

                df = self.df.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

                k1 = 1.5
                b = 0.75
                denom = freq + k1 * (1 - b + b * (dl / (self.avgdl or 1.0)))
                score += idf * (freq * (k1 + 1)) / denom

            scores.append((score, memory_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [memory_id for _, memory_id in scores[:top_k]]

    def decide(
        self,
        user_input: str,
        candidate_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        selected = [self._memory_id(memory) for memory in candidate_memories]

        return {
            "should_inject": bool(selected),
            "selected_memories": selected,
            "reason": "BM25 baseline: always inject all recalled memories.",
        }

    def _build(self) -> None:
        if self.built:
            return

        if self.doc_tokens:
            self.avgdl = sum(len(tokens) for tokens in self.doc_tokens) / len(
                self.doc_tokens
            )
        else:
            self.avgdl = 0.0

        self.built = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _memory_id(memory: dict[str, Any]) -> str:
        for key in ("memory_id", "id", "session_id"):
            if key in memory:
                return str(memory[key])
        raise KeyError(f"Memory has no memory_id/id/session_id: {memory}")

    @staticmethod
    def _memory_text(memory: dict[str, Any]) -> str:
        for key in ("text", "memory", "content", "summary"):
            if key in memory:
                return str(memory[key])
        return json.dumps(memory, ensure_ascii=False)


def main():
    evaluator = MemGateEvaluator("datas/memgate-eval.jsonl")

    # result = evaluator.evaluate(
    #     system_factory=lambda: MySystem(),
    #     top_k=10,
    #     save_records_path="results/runs/my_system.jsonl",
    #     verbose=False,
    #     judge_llm_config=None,
    #     answer_llm_config=None,
    # )
    # result = evaluator.evaluate(
    #     system_factory=lambda: FooSystem(),
    #     top_k=10,
    #     max_samples=3,
    #     save_records_path="results/runs/foo_system.jsonl",
    #     verbose=False,
    #     judge_llm_config=None,
    #     answer_llm_config=None,
    # )
    # result = evaluator.evaluate(
    #     system_factory=lambda: HybridBroadRecallSystem(),
    #     top_k=50,
    #     probe_types=["required", "availability_with_memory"],
    #     save_records_path="results/runs/hybrid_broad_recall_top10.jsonl",
    #     verbose=False,
    # )
    result = evaluator.evaluate(
        system_factory=lambda: HybridBertGateSystem(),
        top_k=10,
        probe_types=["required", "availability_with_memory"],
        save_records_path="results/runs/hybrid_bert_gate_positive_top10.jsonl",
        verbose=False,
    )

    evaluator.print_report(result)


if __name__ == "__main__":
    main()
