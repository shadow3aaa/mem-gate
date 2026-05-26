"""
HybridBertGateSystem for MemGate.

A non-generative memory gate baseline:

    broad recall:  BM25 + optional Ollama bge-m3 dense retrieval
    rerank/gate:   sentence-transformers CrossEncoder / reranker

Install:
    uv add sentence-transformers torch transformers

Optional if you use dense broad recall through Ollama:
    ollama pull bge-m3

Usage in main.py:

    from eval import MemGateEvaluator
    from hybrid_bert_gate_system import HybridBertGateSystem

    def main():
        evaluator = MemGateEvaluator("datas/memgate-eval.jsonl")
        result = evaluator.evaluate(
            system_factory=lambda: HybridBertGateSystem(),
            top_k=10,
            save_records_path="results/runs/hybrid_bert_gate.jsonl",
            verbose=False,
        )
        MemGateEvaluator.print_report(result)

Environment variables:
    OLLAMA_BASE_URL=http://127.0.0.1:11434
    MEMGATE_EMBED_MODEL=bge-m3
    MEMGATE_USE_DENSE=1

    MEMGATE_RERANK_MODEL=BAAI/bge-reranker-base
    MEMGATE_RERANK_DEVICE=auto       # e.g. cuda, cpu, mps, or auto
    MEMGATE_RERANK_BATCH_SIZE=8
    MEMGATE_RERANK_MAX_LENGTH=512
    MEMGATE_RERANK_MAX_CHARS=2400

    MEMGATE_POOL_SIZE=50             # broad recall pool before reranking
    MEMGATE_GATE_THRESHOLD=0.5       # tune on dev, not test

    MEMGATE_DENSE_WEIGHT=0.55
    MEMGATE_BM25_WEIGHT=0.45
    MEMGATE_SELF_BONUS=0.04
    MEMGATE_DEBUG=0

Notes:
    - This class does not call any chat/generative LLM.
    - recall(...) returns CrossEncoder-reranked ids, so MemGate R@k reflects reranking.
    - decide(...) uses the cached CrossEncoder score from recall(...).
    - Threshold should be tuned on a dev set. For pure rerank diagnostics on positive-only
      subsets, you may set MEMGATE_GATE_THRESHOLD=0.0.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MemoryDoc:
    doc_id: str
    memory_id: str
    text: str
    kind: str  # raw | user_turns | self_snippets


class HybridBertGateSystem:
    """
    MemGate-compatible system.

    index(memory_id, memory):
        Store raw memory and build lightweight lexical documents.

    recall(user_input, top_k):
        1. Broad-recall a pool with BM25 + dense embeddings.
        2. CrossEncoder reranks the pool.
        3. Return reranked top_k memory IDs.

    decide(user_input, candidate_memories):
        Use the best CrossEncoder score among recalled candidate_memories.
        If score >= threshold, inject only the best memory; otherwise abstain.
    """

    _embed_cache: dict[str, list[float]] = {}
    _reranker_cache: dict[str, Any] = {}

    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        embed_model: str | None = None,
        rerank_model: str | None = None,
        request_timeout: float = 60.0,
    ) -> None:
        self.ollama_base_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.embed_model = embed_model or os.getenv("MEMGATE_EMBED_MODEL", "bge-m3")
        self.rerank_model = rerank_model or os.getenv(
            "MEMGATE_RERANK_MODEL", "BAAI/bge-reranker-base"
        )
        self.request_timeout = float(
            os.getenv("MEMGATE_REQUEST_TIMEOUT", str(request_timeout))
        )

        self.use_dense = os.getenv("MEMGATE_USE_DENSE", "1") != "0"
        self.pool_size = int(os.getenv("MEMGATE_POOL_SIZE", "50"))
        self.gate_threshold = float(os.getenv("MEMGATE_GATE_THRESHOLD", "0.5"))
        self.rerank_batch_size = int(os.getenv("MEMGATE_RERANK_BATCH_SIZE", "8"))
        self.rerank_max_length = int(os.getenv("MEMGATE_RERANK_MAX_LENGTH", "512"))
        self.rerank_max_chars = int(os.getenv("MEMGATE_RERANK_MAX_CHARS", "2400"))
        self.rerank_device = os.getenv("MEMGATE_RERANK_DEVICE", "auto")

        self.dense_weight = float(os.getenv("MEMGATE_DENSE_WEIGHT", "0.55"))
        self.bm25_weight = float(os.getenv("MEMGATE_BM25_WEIGHT", "0.45"))
        self.self_bonus = float(os.getenv("MEMGATE_SELF_BONUS", "0.04"))
        self.debug = os.getenv("MEMGATE_DEBUG", "0") == "1"

        self.memories_by_id: dict[str, dict[str, Any]] = {}
        self.memory_order: list[str] = []
        self.memory_text_by_id: dict[str, str] = {}
        self.rerank_text_by_id: dict[str, str] = {}
        self.self_snippets_by_id: dict[str, list[str]] = {}

        self.docs: list[MemoryDoc] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_tf: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0
        self._doc_embeddings: np.ndarray | None = None
        self._built = False

        self._last_recall: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # MemGate interface
    # ------------------------------------------------------------------

    def index(self, memory_id: str, memory: dict[str, Any]) -> None:
        self.memories_by_id[memory_id] = memory
        self.memory_order.append(memory_id)

        raw = self._memory_text(memory)
        user_turns = self._user_turn_text(memory)
        self_snippets = self._self_statement_snippets(memory)

        self.memory_text_by_id[memory_id] = raw
        self.self_snippets_by_id[memory_id] = self_snippets
        self.rerank_text_by_id[memory_id] = self._make_rerank_text(
            raw=raw,
            user_turns=user_turns,
            self_snippets=self_snippets,
        )

        doc_texts: list[tuple[str, str]] = []
        doc_texts.append(("raw", self._head_tail(raw, max_chars=3500)))
        if user_turns.strip():
            doc_texts.append(
                ("user_turns", self._head_tail(user_turns, max_chars=3500))
            )
        if self_snippets:
            doc_texts.append(
                (
                    "self_snippets",
                    self._head_tail("\n".join(self_snippets[:24]), max_chars=2600),
                )
            )

        for kind, text in doc_texts:
            if not text.strip():
                continue
            doc = MemoryDoc(
                doc_id=f"{memory_id}::{kind}",
                memory_id=memory_id,
                text=text,
                kind=kind,
            )
            self.docs.append(doc)
            toks = self._tokenize(text)
            self.doc_tokens.append(toks)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            for tok in set(toks):
                self.df[tok] += 1

        self._built = False

    def recall(self, user_input: str, top_k: int) -> list[str]:
        self._build()
        if not self.memory_order:
            return []

        pool_k = max(top_k, self.pool_size)
        broad_ids, broad_scores = self._broad_recall(user_input, pool_k=pool_k)

        reranked = self._rerank(user_input, broad_ids)
        recalled = [mid for mid, _ in reranked[:top_k]]

        self._last_recall = {
            "user_input": user_input,
            "broad_ids": broad_ids,
            "broad_scores": broad_scores,
            "rerank_scores": {mid: score for mid, score in reranked},
            "reranked_ids": [mid for mid, _ in reranked],
            "recalled": recalled,
            "pool_k": pool_k,
        }

        return recalled

    def decide(
        self, user_input: str, candidate_memories: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not candidate_memories:
            return {
                "should_inject": False,
                "selected_memories": [],
                "reason": "no candidates",
            }

        # Evaluator calls decide right after recall. If cache is unavailable for
        # any reason, score the candidate memories directly.
        scores = self._last_recall.get("rerank_scores")
        if (
            not isinstance(scores, dict)
            or self._last_recall.get("user_input") != user_input
        ):
            ids = [self._memory_id(m) for m in candidate_memories]
            scores = {mid: score for mid, score in self._rerank(user_input, ids)}

        best_mid = None
        best_score = -float("inf")
        for memory in candidate_memories:
            mid = self._memory_id(memory)
            score = float(scores.get(mid, -float("inf")))
            if score > best_score:
                best_mid = mid
                best_score = score

        if best_mid is None or best_score < self.gate_threshold:
            return {
                "should_inject": False,
                "selected_memories": [],
                "reason": (
                    f"bert_gate_reject best_score={best_score:.4f} "
                    f"threshold={self.gate_threshold:.4f}"
                ),
            }

        return {
            "should_inject": True,
            "selected_memories": [best_mid],
            "reason": (
                f"bert_gate_select score={best_score:.4f} "
                f"threshold={self.gate_threshold:.4f}"
            ),
        }

    # ------------------------------------------------------------------
    # Broad recall
    # ------------------------------------------------------------------

    def _build(self) -> None:
        if self._built:
            return

        self.avgdl = (
            sum(len(toks) for toks in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 0.0
        )

        if self.use_dense and self.docs:
            embs = [self._embed(doc.text) for doc in self.docs]
            arr = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            self._doc_embeddings = arr / np.maximum(norms, 1e-8)
        else:
            self._doc_embeddings = None

        self._built = True

    def _broad_recall(
        self, user_input: str, *, pool_k: int
    ) -> tuple[list[str], dict[str, float]]:
        query_text = self._query_text(user_input)
        if not self.docs:
            return [], {}

        bm25 = np.array(
            [self._bm25_score(query_text, i) for i in range(len(self.docs))],
            dtype=np.float32,
        )

        if self.use_dense and self._doc_embeddings is not None:
            dense = self._dense_scores(query_text)
            doc_scores = self.bm25_weight * self._minmax(
                bm25
            ) + self.dense_weight * self._minmax(dense)
        else:
            doc_scores = self._minmax(bm25)

        per_mem: dict[str, list[float]] = defaultdict(list)
        kind_bonus: dict[str, float] = defaultdict(float)
        for i, score in enumerate(doc_scores):
            doc = self.docs[i]
            per_mem[doc.memory_id].append(float(score))
            if doc.kind == "self_snippets":
                kind_bonus[doc.memory_id] = max(
                    kind_bonus[doc.memory_id], self.self_bonus
                )
            elif doc.kind == "user_turns":
                kind_bonus[doc.memory_id] = max(
                    kind_bonus[doc.memory_id], self.self_bonus * 0.5
                )

        scored: list[tuple[float, str]] = []
        for mid in self.memory_order:
            scores = sorted(per_mem.get(mid, [0.0]), reverse=True)
            max_score = scores[0]
            top3_mean = sum(scores[:3]) / min(3, len(scores))
            snippet_count_bonus = min(
                0.03, 0.004 * len(self.self_snippets_by_id.get(mid, []))
            )
            final = (
                0.78 * max_score
                + 0.22 * top3_mean
                + kind_bonus[mid]
                + snippet_count_bonus
            )
            scored.append((final, mid))

        scored.sort(key=lambda x: x[0], reverse=True)
        broad = scored[: min(pool_k, len(scored))]
        return [mid for _, mid in broad], {mid: float(score) for score, mid in scored}

    def _dense_scores(self, query_text: str) -> np.ndarray:
        if self._doc_embeddings is None or self._doc_embeddings.size == 0:
            return np.zeros(len(self.docs), dtype=np.float32)
        q = np.array(self._embed(query_text), dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        return self._doc_embeddings @ q

    def _bm25_score(self, query: str, doc_index: int) -> float:
        terms = self._tokenize(query)
        if not terms:
            return 0.0
        n_docs = max(1, len(self.docs))
        tf = self.doc_tf[doc_index]
        dl = len(self.doc_tokens[doc_index]) or 1
        k1 = 1.45
        b = 0.72
        score = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1.0 - b + b * dl / (self.avgdl or 1.0))
            score += idf * (freq * (k1 + 1.0)) / denom
        return float(score)

    # ------------------------------------------------------------------
    # CrossEncoder rerank/gate
    # ------------------------------------------------------------------

    def _rerank(
        self, user_input: str, memory_ids: list[str]
    ) -> list[tuple[str, float]]:
        if not memory_ids:
            return []

        reranker = self._get_reranker()
        pairs = [
            [
                user_input,
                self.rerank_text_by_id.get(mid, self.memory_text_by_id.get(mid, "")),
            ]
            for mid in memory_ids
        ]
        raw_scores = reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

        reranked = list(zip(memory_ids, [float(x) for x in scores]))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _get_reranker(self) -> Any:
        cache_key = f"{self.rerank_model}|{self.rerank_device}|{self.rerank_max_length}"
        cached = HybridBertGateSystem._reranker_cache.get(cache_key)
        if cached is not None:
            return cached

        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {"max_length": self.rerank_max_length}
        if self.rerank_device and self.rerank_device != "auto":
            kwargs["device"] = self.rerank_device

        model = CrossEncoder(self.rerank_model, **kwargs)
        HybridBertGateSystem._reranker_cache[cache_key] = model
        return model

    # ------------------------------------------------------------------
    # Embedding through Ollama
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        text = self._head_tail(text, max_chars=3500)
        key = self._cache_key(self.embed_model, text)
        cached = HybridBertGateSystem._embed_cache.get(key)
        if cached is not None:
            return cached

        try:
            data = self._post_json(
                "/api/embed", {"model": self.embed_model, "input": text}
            )
            embeddings = data.get("embeddings")
            if (
                isinstance(embeddings, list)
                and embeddings
                and isinstance(embeddings[0], list)
            ):
                emb = [float(x) for x in embeddings[0]]
            else:
                raise ValueError("unexpected /api/embed response")
        except Exception:
            data = self._post_json(
                "/api/embeddings", {"model": self.embed_model, "prompt": text}
            )
            emb = [float(x) for x in data["embedding"]]

        HybridBertGateSystem._embed_cache[key] = emb
        return emb

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = self.ollama_base_url + path
        last_err: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed: {url}: {last_err}")

    # ------------------------------------------------------------------
    # Text handling
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_id(memory: dict[str, Any]) -> str:
        for key in ("memory_id", "id", "session_id"):
            if key in memory:
                return str(memory[key])
        raise KeyError(f"memory has no id field: {list(memory.keys())}")

    @staticmethod
    def _memory_text(memory: dict[str, Any]) -> str:
        if "text" in memory:
            return str(memory["text"])
        turns = memory.get("turns")
        if isinstance(turns, list):
            parts = []
            for t in turns:
                if isinstance(t, dict):
                    role = str(t.get("role", ""))
                    content = str(t.get("content", ""))
                    parts.append(f"{role}: {content}")
            return "\n".join(parts)
        for key in ("memory", "content", "summary"):
            if key in memory:
                return str(memory[key])
        clean = {
            k: v
            for k, v in memory.items()
            if k not in {"is_target_memory", "is_forbidden_memory"}
        }
        return json.dumps(clean, ensure_ascii=False)

    @staticmethod
    def _user_turn_text(memory: dict[str, Any]) -> str:
        turns = memory.get("turns")
        if not isinstance(turns, list):
            return ""
        parts = []
        for t in turns:
            if isinstance(t, dict) and str(t.get("role", "")).lower() == "user":
                parts.append(str(t.get("content", "")))
        return "\n".join(parts)

    @staticmethod
    def _self_statement_snippets(memory: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        turns = memory.get("turns")
        if isinstance(turns, list):
            for t in turns:
                if isinstance(t, dict) and str(t.get("role", "")).lower() == "user":
                    texts.append(str(t.get("content", "")))
        else:
            texts.append(HybridBertGateSystem._memory_text(memory))

        first_person = re.compile(
            r"\b(i|i'm|i’ve|i've|i’d|i'd|i’ll|i'll|me|my|mine|we|our|ours)\b",
            re.I,
        )
        signal = re.compile(
            r"\b(live|lived|work|worked|job|role|company|graduated|degree|major|school|study|studied|prefer|preference|favorite|like|own|have|has|use|using|account|email|phone|address|birthday|born|name|pet|dog|cat|car|device|laptop|computer|doctor|allergic|allergy|usually|always|commute|schedule|subscription|plan|read|reading|book|store|budget|birthday|gift)\b",
            re.I,
        )
        out: list[str] = []
        for text in texts:
            for s in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
                s = re.sub(r"\s+", " ", s).strip()
                if (
                    18 <= len(s) <= 650
                    and first_person.search(s)
                    and (signal.search(s) or len(s) <= 220)
                ):
                    out.append(s)

        seen: set[str] = set()
        uniq: list[str] = []
        for s in out:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        return uniq

    def _make_rerank_text(
        self,
        *,
        raw: str,
        user_turns: str,
        self_snippets: list[str],
    ) -> str:
        parts: list[str] = []
        if self_snippets:
            parts.append("USER SELF-STATEMENTS:\n" + "\n".join(self_snippets[:16]))
        if user_turns.strip():
            parts.append("USER TURNS:\n" + self._head_tail(user_turns, max_chars=1200))
        parts.append("RAW MEMORY:\n" + self._head_tail(raw, max_chars=1600))
        return self._head_tail("\n\n".join(parts), max_chars=self.rerank_max_chars)

    @staticmethod
    def _query_text(user_input: str) -> str:
        lowered = user_input.lower()
        expansions = [user_input]

        if any(
            w in lowered
            for w in (
                "blank",
                "missing",
                "fill",
                "field",
                "cell",
                "record",
                "profile",
                "row",
                "entry",
                "value",
                "exact",
                "portal",
                "form",
            )
        ):
            expansions.extend(
                [
                    "current user's own stored personal fact exact value",
                    "durable fact the user previously stated about themselves",
                    "user profile memory self-owned value personal history preference possession account detail",
                    "first person user statement I my me mine value",
                ]
            )
        if any(
            w in lowered
            for w in (
                "degree",
                "education",
                "academic",
                "alumni",
                "school",
                "graduated",
            )
        ):
            expansions.append(
                "education degree graduated academic background school major"
            )
        if any(w in lowered for w in ("pet", "dog", "cat")):
            expansions.append("pet animal dog cat user's pet name")
        if any(w in lowered for w in ("device", "laptop", "phone", "computer")):
            expansions.append("device laptop phone computer user's device")
        if any(w in lowered for w in ("address", "city", "live", "home")):
            expansions.append("address city home where user lives")

        return "\n".join(expansions)

    @staticmethod
    def _head_tail(text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _cache_key(prefix: str, text: str) -> str:
        return prefix + ":" + HybridBertGateSystem._hash(text)
