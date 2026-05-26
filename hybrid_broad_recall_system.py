"""
HybridBroadRecallSystem for MemGate.

Purpose:
  A fast, non-LLM broad-recall baseline. This system is meant to measure
  whether target memories can be pulled into a reasonably wide candidate set
  before adding a BERT/cross-encoder gate.

MemGate interface:
  system.index(memory_id, memory)
  system.recall(user_input, top_k) -> list[str]
  system.decide(user_input, candidate_memories) -> dict

Design:
  - No generative LLM calls.
  - Optional Ollama embedding endpoint for bge-m3 dense retrieval.
  - Manual BM25 over multiple memory views:
      raw memory text
      user-turn text
      self-statement snippets
  - Deterministic query expansion for query-gap prompts.
  - decide() intentionally always injects recalled candidates.
    This is for recall diagnostics, not a safe final memory gate.

Recommended diagnostic usage:
  result = evaluator.evaluate(
      system_factory=lambda: HybridBroadRecallSystem(),
      top_k=10,  # then try 50 / 100 for coverage diagnostics
      probe_types=["required", "availability_with_memory"],
      save_records_path="results/runs/hybrid_broad_recall_top10.jsonl",
  )

Useful env vars:
  OLLAMA_BASE_URL=http://127.0.0.1:11434
  MEMGATE_EMBED_MODEL=bge-m3
  MEMGATE_USE_DENSE=1
  MEMGATE_DENSE_WEIGHT=0.45
  MEMGATE_RAW_BM25_WEIGHT=0.28
  MEMGATE_USER_BM25_WEIGHT=0.22
  MEMGATE_SELF_BM25_WEIGHT=0.25
  MEMGATE_SELF_BONUS_WEIGHT=0.05
  MEMGATE_REQUEST_TIMEOUT=60
  MEMGATE_DEBUG=0
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
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MemoryDoc:
    memory_id: str
    raw_text: str
    user_text: str
    self_text: str
    semantic_text: str
    self_statement_count: int


class _BM25Index:
    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.tokens = [HybridBroadRecallSystem.tokenize(d) for d in docs]
        self.tfs = [Counter(toks) for toks in self.tokens]
        self.df: Counter[str] = Counter()
        for toks in self.tokens:
            for tok in set(toks):
                self.df[tok] += 1
        self.n_docs = max(1, len(docs))
        self.avgdl = (
            sum(len(toks) for toks in self.tokens) / len(self.tokens)
            if self.tokens
            else 0.0
        )

    def scores(self, query: str) -> np.ndarray:
        q_terms = HybridBroadRecallSystem.tokenize(query)
        if not q_terms or not self.docs:
            return np.zeros(len(self.docs), dtype=np.float32)

        k1 = 1.45
        b = 0.72
        scores = np.zeros(len(self.docs), dtype=np.float32)

        for i, tf in enumerate(self.tfs):
            dl = len(self.tokens[i]) or 1
            score = 0.0
            for term in q_terms:
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * (dl / (self.avgdl or 1.0)))
                score += idf * (freq * (k1 + 1.0)) / denom
            scores[i] = score
        return scores


class HybridBroadRecallSystem:
    """
    Fast broad-recall system for MemGate diagnostics.

    This is not intended to be a safe final gate. decide() always injects all
    recalled memories so that Gate=True R@K reflects recall coverage.
    """

    _embed_cache: dict[str, list[float]] = {}

    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        embed_model: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.ollama_base_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.embed_model = embed_model or os.getenv("MEMGATE_EMBED_MODEL", "bge-m3")
        self.request_timeout = float(
            os.getenv("MEMGATE_REQUEST_TIMEOUT", str(request_timeout or 60.0))
        )

        self.use_dense = os.getenv("MEMGATE_USE_DENSE", "1") != "0"
        self.dense_weight = float(os.getenv("MEMGATE_DENSE_WEIGHT", "0.45"))
        self.raw_bm25_weight = float(os.getenv("MEMGATE_RAW_BM25_WEIGHT", "0.28"))
        self.user_bm25_weight = float(os.getenv("MEMGATE_USER_BM25_WEIGHT", "0.22"))
        self.self_bm25_weight = float(os.getenv("MEMGATE_SELF_BM25_WEIGHT", "0.25"))
        self.self_bonus_weight = float(os.getenv("MEMGATE_SELF_BONUS_WEIGHT", "0.05"))
        self.debug = os.getenv("MEMGATE_DEBUG", "0") == "1"

        self.docs: list[MemoryDoc] = []
        self.memories_by_id: dict[str, dict[str, Any]] = {}
        self._built = False

        self._raw_bm25: _BM25Index | None = None
        self._user_bm25: _BM25Index | None = None
        self._self_bm25: _BM25Index | None = None
        self._dense_matrix: np.ndarray | None = None

        # Diagnostic fields. eval.py does not need them, but external scripts can inspect.
        self.last_scores: dict[str, float] = {}
        self.last_broad_pool_ids: list[str] = []
        self.last_query_text: str = ""

    # ------------------------------------------------------------------
    # MemGate interface
    # ------------------------------------------------------------------

    def index(self, memory_id: str, memory: dict[str, Any]) -> None:
        raw_text = self.memory_text(memory)
        user_text = self.user_turn_text(memory)
        self_snippets = self.self_statement_snippets(memory)
        self_text = "\n".join(self_snippets)

        semantic_text = self.build_semantic_text(
            raw_text=raw_text,
            user_text=user_text,
            self_text=self_text,
        )

        self.memories_by_id[memory_id] = memory
        self.docs.append(
            MemoryDoc(
                memory_id=memory_id,
                raw_text=raw_text,
                user_text=user_text,
                self_text=self_text,
                semantic_text=semantic_text,
                self_statement_count=len(self_snippets),
            )
        )
        self._built = False

    def recall(self, user_input: str, top_k: int) -> list[str]:
        self._build()
        if not self.docs:
            return []

        query_text = self.query_expansion(user_input)
        self.last_query_text = query_text

        raw_scores = self._raw_bm25.scores(query_text) if self._raw_bm25 else self._zeros()
        user_scores = self._user_bm25.scores(query_text) if self._user_bm25 else self._zeros()
        self_scores = self._self_bm25.scores(query_text) if self._self_bm25 else self._zeros()

        total = (
            self.raw_bm25_weight * self.minmax(raw_scores)
            + self.user_bm25_weight * self.minmax(user_scores)
            + self.self_bm25_weight * self.minmax(self_scores)
        )

        if self.use_dense:
            try:
                dense_scores = self._dense_scores(query_text)
                total += self.dense_weight * self.minmax(dense_scores)
            except Exception as exc:
                if self.debug:
                    print(f"[HybridBroadRecallSystem] dense retrieval disabled for this query: {exc}")

        self_bonus = np.array(
            [min(1.0, doc.self_statement_count / 4.0) for doc in self.docs],
            dtype=np.float32,
        )
        total += self.self_bonus_weight * self_bonus

        # Tiny stable tie-breaker: preserve insertion order deterministically.
        total += np.array(
            [1e-9 * (len(self.docs) - i) for i in range(len(self.docs))],
            dtype=np.float32,
        )

        order = np.argsort(-total)
        ranked_ids = [self.docs[int(i)].memory_id for i in order[:top_k]]
        self.last_broad_pool_ids = ranked_ids
        self.last_scores = {self.docs[int(i)].memory_id: float(total[int(i)]) for i in order}
        return ranked_ids

    def decide(
        self,
        user_input: str,
        candidate_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Diagnostic-only behavior: always inject every recalled memory.
        # This makes Gate=True R@K mostly reflect recall coverage. Negative probes
        # will intentionally fail, which is expected for this baseline.
        selected = [self.memory_id(memory) for memory in candidate_memories]
        return {
            "should_inject": bool(selected),
            "selected_memories": selected,
            "reason": "HybridBroadRecallSystem diagnostic baseline: always inject recalled memories.",
        }

    # ------------------------------------------------------------------
    # Build / scoring
    # ------------------------------------------------------------------

    def _build(self) -> None:
        if self._built:
            return

        self._raw_bm25 = _BM25Index([doc.raw_text for doc in self.docs])
        self._user_bm25 = _BM25Index([doc.user_text for doc in self.docs])
        self._self_bm25 = _BM25Index([doc.self_text for doc in self.docs])

        if self.use_dense and self.docs:
            embs = [self.embed(doc.semantic_text) for doc in self.docs]
            arr = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            self._dense_matrix = arr / np.maximum(norms, 1e-8)
        else:
            self._dense_matrix = None

        self._built = True

    def _dense_scores(self, query_text: str) -> np.ndarray:
        if self._dense_matrix is None or self._dense_matrix.size == 0:
            return self._zeros()
        q = np.array(self.embed(query_text), dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        return self._dense_matrix @ q

    def _zeros(self) -> np.ndarray:
        return np.zeros(len(self.docs), dtype=np.float32)

    @staticmethod
    def minmax(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo < 1e-8:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    @staticmethod
    def query_expansion(user_input: str) -> str:
        """Deterministic broad expansion. Keep generic; avoid dataset IDs/labels."""
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
                "sheet",
                "template",
            )
        ):
            expansions.extend(
                [
                    "current user's own stored personal fact exact value",
                    "durable fact the user previously stated about themselves",
                    "first person user statement I my me mine personal value",
                    "user profile memory preference possession account detail history",
                ]
            )

        # Broad domain hints. These are intentionally general and not a fixed answer key.
        domain_hints = [
            (("degree", "education", "academic", "alumni", "school", "graduated", "major"),
             "education degree school graduated academic background major"),
            (("pet", "dog", "cat", "animal"),
             "pet animal dog cat user's pet"),
            (("device", "laptop", "phone", "computer", "camera"),
             "device laptop phone computer user owns uses"),
            (("address", "city", "live", "home", "location"),
             "address city home location where user lives"),
            (("commute", "morning", "schedule", "routine"),
             "routine schedule commute morning habit"),
            (("food", "recipe", "snack", "coffee", "restaurant"),
             "food recipe snack coffee restaurant preference habit"),
            (("book", "movie", "music", "show", "author"),
             "book movie music show author preference"),
            (("work", "job", "company", "career", "coworker"),
             "work job company career professional history"),
        ]
        for triggers, hint in domain_hints:
            if any(t in lowered for t in triggers):
                expansions.append(hint)

        return "\n".join(expansions)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        text = self.head_tail(text, max_chars=3500)
        key = self.cache_key(self.embed_model, text)
        cached = HybridBroadRecallSystem._embed_cache.get(key)
        if cached is not None:
            return cached

        # Ollama has both /api/embed and older /api/embeddings.
        try:
            data = self.post_json("/api/embed", {"model": self.embed_model, "input": text})
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                emb = [float(x) for x in embeddings[0]]
            else:
                raise ValueError("unexpected /api/embed response")
        except Exception:
            data = self.post_json("/api/embeddings", {"model": self.embed_model, "prompt": text})
            emb = [float(x) for x in data["embedding"]]

        HybridBroadRecallSystem._embed_cache[key] = emb
        return emb

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
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
            ) as exc:
                last_err = exc
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed: {url}: {last_err}")

    # ------------------------------------------------------------------
    # Memory text handling
    # ------------------------------------------------------------------

    @staticmethod
    def memory_id(memory: dict[str, Any]) -> str:
        for key in ("memory_id", "id", "session_id"):
            if key in memory:
                return str(memory[key])
        raise KeyError(f"memory has no id field: {list(memory.keys())}")

    @staticmethod
    def memory_text(memory: dict[str, Any]) -> str:
        if "text" in memory:
            return str(memory["text"])
        turns = memory.get("turns")
        if isinstance(turns, list):
            parts = []
            for turn in turns:
                if isinstance(turn, dict):
                    role = str(turn.get("role", ""))
                    content = str(turn.get("content", ""))
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
    def user_turn_text(memory: dict[str, Any]) -> str:
        turns = memory.get("turns")
        if isinstance(turns, list):
            out = []
            for turn in turns:
                if isinstance(turn, dict) and str(turn.get("role", "")).lower() == "user":
                    out.append(str(turn.get("content", "")))
            return "\n".join(out)
        return HybridBroadRecallSystem.memory_text(memory)

    @staticmethod
    def self_statement_snippets(memory: dict[str, Any]) -> list[str]:
        text = HybridBroadRecallSystem.user_turn_text(memory)
        first_person = re.compile(
            r"\b(i|i'm|i’ve|i've|i’d|i'd|i’ll|i'll|me|my|mine|we|our|ours)\b",
            re.I,
        )
        signal = re.compile(
            r"\b(live|lived|work|worked|job|role|company|graduated|degree|major|school|study|studied|prefer|preference|favorite|like|own|have|has|use|using|account|email|phone|address|birthday|born|name|pet|dog|cat|car|device|laptop|computer|doctor|allergic|allergy|usually|always|commute|schedule|subscription|plan|remember|bought|visited|moved|changed)\b",
            re.I,
        )

        snippets: list[str] = []
        for sent in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
            sent = re.sub(r"\s+", " ", sent).strip()
            if not (18 <= len(sent) <= 700):
                continue
            if first_person.search(sent) and (signal.search(sent) or len(sent) <= 240):
                snippets.append(sent)

        # Stable de-duplication.
        seen: set[str] = set()
        out: list[str] = []
        for s in snippets:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    @staticmethod
    def build_semantic_text(*, raw_text: str, user_text: str, self_text: str) -> str:
        parts = []
        if self_text.strip():
            parts.append("USER SELF-STATEMENTS:\n" + self_text)
        if user_text.strip():
            parts.append("USER TURNS:\n" + HybridBroadRecallSystem.head_tail(user_text, max_chars=1800))
        parts.append("RAW MEMORY:\n" + HybridBroadRecallSystem.head_tail(raw_text, max_chars=2200))
        return "\n\n".join(parts)

    @staticmethod
    def head_tail(text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...\n" + text[-half:]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def cache_key(prefix: str, text: str) -> str:
        return prefix + ":" + HybridBroadRecallSystem.hash_text(text)
