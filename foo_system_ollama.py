"""
FooSystem for MemGate: edge-style memory gate baseline using local Ollama.

Models expected in `ollama ls`:
  - bge-m3:latest          for embeddings
  - qwen3.5:0.8b           for scope gate / fact extraction / verifier

Drop this file into the mem-gate repo, then in main.py:

    from foo_system_ollama import FooSystem
    ...
    result = evaluator.evaluate(system_factory=lambda: FooSystem(), top_k=10, ...)

No Ollama SDK dependency; uses urllib against http://127.0.0.1:11434.

Useful env vars:
  OLLAMA_BASE_URL=http://127.0.0.1:11434
  MEMGATE_EMBED_MODEL=bge-m3
  MEMGATE_LLM=qwen3.5:0.8b
  MEMGATE_USE_FACT_EXTRACTION=1        # set 0 for faster debug
  MEMGATE_EXTRACT_MAX_FACTS=8
  MEMGATE_VERIFY_TOP_N=8
  MEMGATE_SCOPE_THRESHOLD=0.50
  MEMGATE_VERIFY_THRESHOLD=0.62
  MEMGATE_DENSE_WEIGHT=0.58
  MEMGATE_INCLUDE_RAW_FALLBACK=1
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from json_repair import repair_json


@dataclass
class FactCard:
    fact_id: str
    source_memory_id: str
    text: str
    value: str = ""
    predicate: str = ""
    evidence_span: str = ""
    owner: str = "current_user"
    kind: str = "extracted"  # extracted | snippet | raw_fallback


class ModelOutputFormatError(RuntimeError):
    """Raised when the local LLM returns invalid JSON or an invalid schema."""


class FooSystem:
    """
    Edge-style MemGate system:
      1) index-time fact cache from candidate memories
      2) hybrid BM25 + bge-m3 retrieval over fact cards
      3) qwen3.5:0.8b scope gate
      4) qwen3.5:0.8b batched pairwise verifier

    Important design rule:
      Retrieval is allowed to recall hard negatives. Injection is decided only by:
        scope_allowed(user_input) AND verified_evidence(candidate_fact)
    """

    # Class-level caches survive MemGate evaluator's per-sample system recreation.
    _embed_cache: dict[str, list[float]] = {}
    _chat_cache: dict[str, str] = {}
    _fact_cache_by_text_hash: dict[str, list[dict[str, Any]]] = {}
    _compact_cache: dict[str, str] = {}

    def __init__(
        self,
        *,
        ollama_base_url: str | None = None,
        embed_model: str | None = None,
        llm_model: str | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        self.ollama_base_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        ).rstrip("/")
        self.embed_model = embed_model or os.getenv("MEMGATE_EMBED_MODEL", "bge-m3")
        self.llm_model = llm_model or os.getenv("MEMGATE_LLM", "qwen3.5:0.8b")
        self.request_timeout = request_timeout

        self.use_fact_extraction = os.getenv("MEMGATE_USE_FACT_EXTRACTION", "1") != "0"
        self.include_raw_fallback = (
            os.getenv("MEMGATE_INCLUDE_RAW_FALLBACK", "1") != "0"
        )
        self.extract_max_facts = int(os.getenv("MEMGATE_EXTRACT_MAX_FACTS", "8"))
        self.verify_top_n = int(os.getenv("MEMGATE_VERIFY_TOP_N", "8"))
        self.scope_threshold = float(os.getenv("MEMGATE_SCOPE_THRESHOLD", "0.50"))
        self.verify_threshold = float(os.getenv("MEMGATE_VERIFY_THRESHOLD", "0.62"))
        self.dense_weight = float(os.getenv("MEMGATE_DENSE_WEIGHT", "0.58"))
        self.debug = os.getenv("MEMGATE_DEBUG", "0") == "1"

        # Per-evaluation-record state.
        self.memories_by_id: dict[str, dict[str, Any]] = {}
        self.memory_order: list[str] = []
        self.cards: list[FactCard] = []
        self.cards_by_memory_id: dict[str, list[FactCard]] = defaultdict(list)

        self.card_tokens: list[list[str]] = []
        self.card_tf: list[Counter[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0
        self._card_embeddings: np.ndarray | None = None
        self._built = False

        # Used by decide to avoid recomputing retrieval scores when possible.
        self._last_recall: dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # MemGate interface
    # ---------------------------------------------------------------------

    def index(self, memory_id: str, memory: dict[str, Any]) -> None:
        self.memories_by_id[memory_id] = memory
        self.memory_order.append(memory_id)

        for card in self._cards_for_memory(memory_id, memory):
            self.cards.append(card)
            self.cards_by_memory_id[memory_id].append(card)
            toks = self._tokenize(card.text)
            self.card_tokens.append(toks)
            self.card_tf.append(Counter(toks))
            for tok in set(toks):
                self.df[tok] += 1

        self._built = False

    def recall(self, user_input: str, top_k: int) -> list[str]:
        self._build()
        if not self.memory_order:
            return []

        query_bundle = self._query_bundle(user_input)
        query_text = query_bundle["query_text"]

        card_scores = self._hybrid_card_scores(query_text)

        # Aggregate card scores into memory scores. Max catches one strong fact; mean top-3
        # helps memories containing several moderately relevant snippets.
        per_mem: dict[str, list[float]] = defaultdict(list)
        for i, score in enumerate(card_scores):
            per_mem[self.cards[i].source_memory_id].append(float(score))

        memory_scores: list[tuple[float, str]] = []
        for mid in self.memory_order:
            scores = sorted(per_mem.get(mid, [0.0]), reverse=True)
            max_score = scores[0]
            top3_mean = sum(scores[:3]) / min(3, len(scores))
            # tiny explicit self-fact bonus helps generic query-gap prompts without using labels
            self_fact_bonus = min(
                0.04,
                0.008
                * len(
                    [
                        c
                        for c in self.cards_by_memory_id.get(mid, [])
                        if c.kind in {"extracted", "snippet"}
                    ]
                ),
            )
            final = 0.78 * max_score + 0.22 * top3_mean + self_fact_bonus
            memory_scores.append((final, mid))

        memory_scores.sort(key=lambda x: x[0], reverse=True)
        recalled = [mid for _, mid in memory_scores[:top_k]]
        self._last_recall = {
            "user_input": user_input,
            "query_bundle": query_bundle,
            "memory_scores": dict(memory_scores),
            "recalled": recalled,
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

        scope = self._scope_gate(user_input)
        if not scope.get("memory_allowed", False):
            return {
                "should_inject": False,
                "selected_memories": [],
                "reason": f"scope_gate_reject: {self._brief(scope)}",
            }

        # Candidate ids from evaluator are exactly recalled ids. Verification must select
        # only among these, otherwise evaluator marks selected_not_recalled.
        candidate_ids = [self._memory_id(m) for m in candidate_memories]

        # Build verifier candidates from best fact cards of each recalled memory.
        verifier_items: list[tuple[str, FactCard]] = []
        for mid in candidate_ids:
            best_cards = self._best_cards_for_verifier(user_input, mid, max_cards=2)
            for card in best_cards:
                verifier_items.append((mid, card))

        # Keep verifier small for 0.8B. Preserve memory order but prefer stronger card scores.
        verifier_items = verifier_items[: max(1, self.verify_top_n)]
        if not verifier_items:
            return {
                "should_inject": False,
                "selected_memories": [],
                "reason": "scope_allowed_but_no_fact_cards",
            }

        verdicts = self._verify_batch(user_input, scope, verifier_items)

        accepted_by_mem: dict[str, tuple[float, dict[str, Any], FactCard]] = {}
        for (mid, card), verdict in zip(verifier_items, verdicts):
            score = self._to_float(verdict.get("score"), default=0.0)
            ok = (
                bool(verdict.get("can_inject", False))
                and bool(verdict.get("owner_match", True))
                and bool(verdict.get("evidence_sufficient", False))
                and bool(verdict.get("privacy_scope_ok", True))
                and score >= self.verify_threshold
            )
            if not ok:
                continue
            prev = accepted_by_mem.get(mid)
            if prev is None or score > prev[0]:
                accepted_by_mem[mid] = (score, verdict, card)

        if not accepted_by_mem:
            return {
                "should_inject": False,
                "selected_memories": [],
                "reason": f"scope_allowed_but_no_verified_evidence: {self._brief(scope)}",
            }

        ranked = sorted(accepted_by_mem.items(), key=lambda kv: kv[1][0], reverse=True)
        top_mid, (top_score, top_verdict, top_card) = ranked[0]

        # Optional conservative margin. Do not require it when only one accepted.
        if len(ranked) >= 2:
            second_score = ranked[1][1][0]
            if (
                top_score - second_score < 0.04
                and top_score < self.verify_threshold + 0.10
            ):
                return {
                    "should_inject": False,
                    "selected_memories": [],
                    "reason": f"ambiguous_verified_evidence top={top_score:.2f} second={second_score:.2f}",
                }

        return {
            "should_inject": True,
            "selected_memories": [top_mid],
            "reason": (
                f"selected_by_scope+pairwise_verifier score={top_score:.3f}; "
                f"card_kind={top_card.kind}; value={top_card.value[:80]!r}; verdict={self._brief(top_verdict)}"
            ),
        }

    # ---------------------------------------------------------------------
    # Index-time fact cache
    # ---------------------------------------------------------------------

    def _cards_for_memory(
        self, memory_id: str, memory: dict[str, Any]
    ) -> list[FactCard]:
        raw = self._memory_text(memory)
        compact = self._compact_memory_text(memory, max_chars=5000)
        text_hash = self._hash(raw)

        cards: list[FactCard] = []
        if self.use_fact_extraction:
            if text_hash in FooSystem._fact_cache_by_text_hash:
                fact_objs = FooSystem._fact_cache_by_text_hash[text_hash]
            else:
                fact_objs = self._extract_facts(compact)
                FooSystem._fact_cache_by_text_hash[text_hash] = fact_objs

            for j, f in enumerate(fact_objs[: self.extract_max_facts]):
                value = str(f.get("value", "")).strip()
                predicate = str(f.get("predicate", "")).strip()
                evidence = str(f.get("evidence_span", "")).strip()
                fact_text = str(f.get("fact", "")).strip()
                if not fact_text:
                    fact_text = f"The user {predicate}: {value}. Evidence: {evidence}"
                text = self._card_text(
                    owner="current_user",
                    predicate=predicate,
                    value=value,
                    fact=fact_text,
                    evidence=evidence,
                    raw_hint="",
                )
                cards.append(
                    FactCard(
                        fact_id=f"{memory_id}::fact::{j}",
                        source_memory_id=memory_id,
                        text=text,
                        value=value,
                        predicate=predicate,
                        evidence_span=evidence,
                        owner="current_user",
                        kind="extracted",
                    )
                )

        # Cheap fallback: first-person self-statement snippets from user turns. This is often
        # enough for LongMemEval-derived facts, and it protects recall when 0.8B extraction misses.
        snippets = self._self_statement_snippets(memory)
        for j, snip in enumerate(snippets[:10]):
            text = self._card_text(
                owner="current_user_possible",
                predicate="self_statement",
                value="",
                fact=snip,
                evidence=snip,
                raw_hint="",
            )
            cards.append(
                FactCard(
                    fact_id=f"{memory_id}::snippet::{j}",
                    source_memory_id=memory_id,
                    text=text,
                    evidence_span=snip,
                    owner="current_user_possible",
                    kind="snippet",
                )
            )

        # Raw fallback keeps recall robust for cases not captured by fact extraction.
        if self.include_raw_fallback or not cards:
            cards.append(
                FactCard(
                    fact_id=f"{memory_id}::raw",
                    source_memory_id=memory_id,
                    text=self._card_text(
                        owner="unknown",
                        predicate="raw_memory_context",
                        value="",
                        fact=self._head_tail(compact, max_chars=2200),
                        evidence="",
                        raw_hint="raw fallback; verifier must be conservative",
                    ),
                    kind="raw_fallback",
                )
            )

        return cards

    def _extract_facts(self, compact_memory_text: str) -> list[dict[str, Any]]:
        prompt = f"""
Extract compact atomic durable facts about the CURRENT USER from the memory text.
Return ONLY JSON with this schema:
{{
  "facts": [
    {{
      "predicate": "short_relation_name",
      "value": "",
      "fact": "short sentence about the user",
      "evidence_span": "optional short supporting span"
    }}
  ]
}}

Rules:
- Extract only durable real-world facts about the current user.
- Prefer explicit first-person statements: I/my/me/we/our.
- Extract stable personal facts, preferences, possessions, account details, dates, places, education, work, habits, constraints.
- Do not extract assistant advice, examples, puzzles, story/fiction/roleplay content, character relationships, or third-party facts.
- Do not extract meta facts about the text, wording, first-person usage, the user asking a question, or the conversation format.
- Do not extract temporary reactions, thanks, excitement, curiosity, "I hope..." statements, or conversational acknowledgements.
- Do not invent values. If no exact compact value exists, set "value" to "" or omit it.
- Keep at most {self.extract_max_facts} facts.
- Each fact must be <= 18 words. Each evidence_span must be <= 20 words.
- Use single quotes inside JSON strings if you need to mention a title. Do not put unescaped double quotes inside string values.
- If there are no durable current-user facts, return {{"facts": []}}.

Memory text:
{compact_memory_text}
""".strip()
        raw = self._chat_json(prompt, cache_prefix="extract")
        obj = self._parse_json_or_raise(
            raw,
            context="extract_facts",
            expected_schema='{ "facts": [ { "predicate": str, "value"?: str, "fact"?: str, "evidence_span"?: str } ] }',
        )

        if not isinstance(obj, dict):
            self._fail_bad_model_output(
                context="extract_facts",
                message=f"Top-level JSON must be an object, got {type(obj).__name__}",
                raw=raw,
            )

        if "facts" not in obj:
            self._fail_bad_model_output(
                context="extract_facts",
                message="Missing required key: facts",
                raw=raw,
            )

        facts = obj["facts"]
        if not isinstance(facts, list):
            self._fail_bad_model_output(
                context="extract_facts",
                message=f"facts must be a list, got {type(facts).__name__}",
                raw=raw,
            )

        clean: list[dict[str, Any]] = []
        for i, f in enumerate(facts):
            if not isinstance(f, dict):
                self._fail_bad_model_output(
                    context="extract_facts",
                    message=f"facts[{i}] must be an object, got {type(f).__name__}",
                    raw=raw,
                )

            # Extraction is an index-time enrichment step, so single fact objects
            # are intentionally tolerant: optional fields are normalized to "".
            # The top-level JSON shape remains strict, while malformed individual
            # fact fields are filtered rather than crashing the whole evaluation.
            predicate = str(f.get("predicate", "")).strip()
            value = str(f.get("value", "")).strip()
            fact = str(f.get("fact", "")).strip()
            evidence = str(f.get("evidence_span", "")).strip()

            # Conservative: require at least one useful text-bearing field.
            if not (fact or evidence or value):
                continue

            if self._is_bad_extracted_fact(
                predicate=predicate,
                value=value,
                fact=fact,
                evidence=evidence,
            ):
                continue

            clean.append(
                {
                    "predicate": predicate,
                    "value": value,
                    "fact": fact,
                    "evidence_span": evidence,
                }
            )

        return clean[: self.extract_max_facts]

    @staticmethod
    def _is_bad_extracted_fact(
        *,
        predicate: str,
        value: str,
        fact: str,
        evidence: str,
    ) -> bool:
        """Filter common LLM extraction artifacts before they become FactCards."""
        core = " ".join([predicate, value, fact]).lower()
        all_text = " ".join([predicate, value, fact, evidence]).lower()

        # Meta facts about the text/conversation are not durable user memories.
        core_bad_phrases = (
            "first-person statement",
            "first person statement",
            "provided text",
            "conversation format",
            "wording",
            "the user uses",
            "uses first-person",
            "uses first person",
            "the user asks",
            "the user asked",
            "the user requests",
            "the user requested",
            "the user wants to know",
            "the user is asking",
            "the user is curious about the theme",
            "describe ",
            "relationship with",
            "chapter by chapter",
            "story",
            "fiction",
            "roleplay",
            "character",
            "plot",
            "assistant advice",
        )
        if any(p in core for p in core_bad_phrases):
            return True

        # Common assistant/conversation artifacts observed in local small-model outputs.
        all_bad_phrases = (
            "you are open to",
            "better sense of",
            "hope this gives you",
            "thanks for",
            "glad you",
            "excited to hear",
            "excited to try",
            "i hope this",
            "as an assistant",
            "assistant:",
            "user self-statement snippets:",
        )
        # Do not reject solely because evidence contains the prefix inserted by
        # _compact_memory_text; reject only when the extracted fact itself is also
        # clearly conversational/meta.
        if any(p in core for p in all_bad_phrases):
            return True

        if re.search(
            r"\b(user|speaker)\s+(uses|asks|asked|requests|requested|mentions|mentioned|says|said|describes)\b",
            core,
        ):
            return True

        # If the fact is just a title/quote extraction with no user relation,
        # it is usually not useful memory evidence.
        if not fact and value and len(value.split()) <= 2 and not predicate:
            return True

        return False

    @staticmethod
    def _card_text(
        *,
        owner: str,
        predicate: str,
        value: str,
        fact: str,
        evidence: str,
        raw_hint: str,
    ) -> str:
        parts = [
            f"owner: {owner}",
            f"predicate: {predicate}",
        ]
        if value:
            parts.append(f"value: {value}")
        if fact:
            parts.append(f"fact: {fact}")
        if evidence:
            parts.append(f"evidence: {evidence}")
        if raw_hint:
            parts.append(f"note: {raw_hint}")
        return "\n".join(parts)

    # ---------------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------------

    def _build(self) -> None:
        if self._built:
            return
        self.avgdl = (
            (sum(len(t) for t in self.card_tokens) / len(self.card_tokens))
            if self.card_tokens
            else 0.0
        )
        if not self.cards:
            self._card_embeddings = np.zeros((0, 1), dtype=np.float32)
            self._built = True
            return
        embs = [self._embed(card.text) for card in self.cards]
        arr = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        self._card_embeddings = arr / np.maximum(norms, 1e-8)
        self._built = True

    def _hybrid_card_scores(self, query_text: str) -> np.ndarray:
        if not self.cards:
            return np.zeros(0, dtype=np.float32)
        bm25 = np.array(
            [self._bm25_score(query_text, i) for i in range(len(self.cards))],
            dtype=np.float32,
        )
        dense = self._dense_scores(query_text)
        return self.dense_weight * self._minmax(dense) + (
            1.0 - self.dense_weight
        ) * self._minmax(bm25)

    def _dense_scores(self, query_text: str) -> np.ndarray:
        assert self._card_embeddings is not None
        if self._card_embeddings.size == 0:
            return np.zeros(0, dtype=np.float32)
        q = np.array(self._embed(query_text), dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        return self._card_embeddings @ q

    def _bm25_score(self, query: str, card_index: int) -> float:
        terms = self._tokenize(query)
        if not terms:
            return 0.0
        n_docs = max(1, len(self.cards))
        tf = self.card_tf[card_index]
        dl = len(self.card_tokens[card_index]) or 1
        score = 0.0
        k1 = 1.45
        b = 0.72
        for term in terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + k1 * (1.0 - b + b * (dl / (self.avgdl or 1.0)))
            score += idf * (freq * (k1 + 1.0)) / denom
        return float(score)

    def _query_bundle(self, user_input: str) -> dict[str, Any]:
        # Fast deterministic expansion. Avoid benchmark-specific slot maps.
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
                    "a durable fact the user previously stated about themselves",
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
        return {"query_text": "\n".join(expansions)}

    def _best_cards_for_verifier(
        self, user_input: str, memory_id: str, max_cards: int = 2
    ) -> list[FactCard]:
        cards = self.cards_by_memory_id.get(memory_id, [])
        if not cards:
            return []
        query_text = self._query_bundle(user_input)["query_text"]
        q_emb = np.array(self._embed(query_text), dtype=np.float32)
        q_emb = q_emb / max(float(np.linalg.norm(q_emb)), 1e-8)
        scored: list[tuple[float, int, FactCard]] = []
        for local_idx, card in enumerate(cards):
            # Use cached embed for each card; only few cards here.
            c_emb = np.array(self._embed(card.text), dtype=np.float32)
            c_emb = c_emb / max(float(np.linalg.norm(c_emb)), 1e-8)
            dense = float(c_emb @ q_emb)
            lexical = self._simple_overlap(query_text, card.text)
            kind_bonus = (
                0.04
                if card.kind == "extracted"
                else (0.02 if card.kind == "snippet" else 0.0)
            )
            scored.append((0.7 * dense + 0.3 * lexical + kind_bonus, local_idx, card))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, _, c in scored[:max_cards]]

    # ---------------------------------------------------------------------
    # Scope gate and pairwise verifier
    # ---------------------------------------------------------------------

    def _scope_gate(self, user_input: str) -> dict[str, Any]:
        prompt = f"""
You are a memory-use scope classifier for a personal assistant.
Decide whether the assistant is allowed to use the CURRENT USER'S private/personal memory for this request.

Return ONLY JSON:
{{
  "memory_allowed": true/false,
  "requested_owner": "current_user" | "third_party" | "generic" | "ambiguous",
  "context_type": "real_user_task" | "mockup" | "placeholder" | "quote" | "demo" | "fictional" | "unknown",
  "requires_exact_user_value": true/false,
  "confidence": 0.0,
  "reason": "short"
}}

Rules:
- Allow only when the request asks for the current user's own real missing/stored value.
- Reject sample data, mockups, placeholders, demos, fictional cases, and named or quoted third-party ownership.
- If a named person wrote "my ...", that "my" belongs to that person, not the current user.
- If ownership is unclear, be conservative and set memory_allowed=false.
- Do not answer the task; classify memory-use permission only.

Request:
{user_input}
""".strip()
        raw = self._chat_json(prompt, cache_prefix="scope")
        obj = self._parse_json_or_raise(
            raw,
            context="scope_gate",
            expected_schema='{ "memory_allowed": bool, "requested_owner": str, "context_type": str, "requires_exact_user_value": bool, "confidence": number, "reason": str }',
        )

        if not isinstance(obj, dict):
            self._fail_bad_model_output(
                context="scope_gate",
                message=f"Top-level JSON must be an object, got {type(obj).__name__}",
                raw=raw,
            )

        required_keys = (
            "memory_allowed",
            "requested_owner",
            "context_type",
            "requires_exact_user_value",
            "confidence",
            "reason",
        )
        for key in required_keys:
            if key not in obj:
                self._fail_bad_model_output(
                    context="scope_gate",
                    message=f"Missing required key: {key}",
                    raw=raw,
                )

        if not isinstance(obj["memory_allowed"], bool):
            self._fail_bad_model_output(
                context="scope_gate",
                message=f"memory_allowed must be bool, got {type(obj['memory_allowed']).__name__}",
                raw=raw,
            )

        if not isinstance(obj["requires_exact_user_value"], bool):
            self._fail_bad_model_output(
                context="scope_gate",
                message=f"requires_exact_user_value must be bool, got {type(obj['requires_exact_user_value']).__name__}",
                raw=raw,
            )

        conf = self._to_float(obj.get("confidence"), default=float("nan"))
        if math.isnan(conf):
            self._fail_bad_model_output(
                context="scope_gate",
                message=f"confidence must be numeric, got {obj.get('confidence')!r}",
                raw=raw,
            )

        owner = str(obj.get("requested_owner", "ambiguous"))
        context = str(obj.get("context_type", "unknown"))

        allowed_owners = {"current_user", "third_party", "generic", "ambiguous"}
        allowed_contexts = {
            "real_user_task",
            "mockup",
            "placeholder",
            "quote",
            "demo",
            "fictional",
            "unknown",
        }
        if owner not in allowed_owners:
            self._fail_bad_model_output(
                context="scope_gate",
                message=f"requested_owner has invalid value: {owner!r}",
                raw=raw,
            )
        if context not in allowed_contexts:
            self._fail_bad_model_output(
                context="scope_gate",
                message=f"context_type has invalid value: {context!r}",
                raw=raw,
            )

        allowed = bool(obj["memory_allowed"]) and conf >= self.scope_threshold

        # A small deterministic safety rail, phrased as generic semantic markers.
        # This is not used for selecting target memories; it only prevents obvious leakage.
        unsafe_contexts = {"mockup", "placeholder", "quote", "demo", "fictional"}
        if (
            owner in {"third_party", "generic", "ambiguous"}
            or context in unsafe_contexts
        ):
            allowed = False

        obj["memory_allowed"] = allowed
        obj["confidence"] = conf
        obj.setdefault("requested_owner", owner)
        obj.setdefault("context_type", context)
        return obj

    def _verify_batch(
        self,
        user_input: str,
        scope: dict[str, Any],
        verifier_items: list[tuple[str, FactCard]],
    ) -> list[dict[str, Any]]:
        item_blocks = []
        for i, (mid, card) in enumerate(verifier_items):
            item_blocks.append(
                f"[{i}] source_memory_id={mid}\n"
                f"fact_kind={card.kind}\n"
                f"{self._head_tail(card.text, max_chars=1200)}"
            )

        prompt = f"""
You are a strict pairwise memory evidence verifier.
For EACH candidate fact, decide whether that candidate alone is safe and sufficient to inject as memory for the request.

Return ONLY JSON:
{{
  "verdicts": [
    {{
      "index": 0,
      "can_inject": true/false,
      "owner_match": true/false,
      "slot_match": true/false,
      "evidence_sufficient": true/false,
      "privacy_scope_ok": true/false,
      "score": 0.0,
      "reason": "short"
    }}
  ]
}}

Strict criteria:
- can_inject=true only if the candidate is about the CURRENT USER and contains a concrete value that directly satisfies the request.
- Related but non-answering facts are false.
- If the request is about a real current-user missing field but this candidate does not identify the exact value, false.
- Do not use memories for samples, mockups, placeholders, quotes, or third parties.
- Prefer precision. A false injection is worse than abstaining.

Request:
{user_input}

Scope frame:
{json.dumps(scope, ensure_ascii=False)}

Candidate facts:
{chr(10).join(item_blocks)}
""".strip()
        raw = self._chat_json(prompt, cache_prefix="verify")
        obj = self._parse_json_or_raise(
            raw,
            context="verify_batch",
            expected_schema='{ "verdicts": [ { "index": int, "can_inject": bool, "owner_match": bool, "slot_match": bool, "evidence_sufficient": bool, "privacy_scope_ok": bool, "score": number, "reason": str } ] }',
        )

        if not isinstance(obj, dict):
            self._fail_bad_model_output(
                context="verify_batch",
                message=f"Top-level JSON must be an object, got {type(obj).__name__}",
                raw=raw,
            )

        if "verdicts" not in obj:
            self._fail_bad_model_output(
                context="verify_batch",
                message="Missing required key: verdicts",
                raw=raw,
            )

        verdicts_raw = obj["verdicts"]
        if not isinstance(verdicts_raw, list):
            self._fail_bad_model_output(
                context="verify_batch",
                message=f"verdicts must be a list, got {type(verdicts_raw).__name__}",
                raw=raw,
            )

        by_idx: dict[int, dict[str, Any]] = {}
        required_bool_keys = (
            "can_inject",
            "owner_match",
            "slot_match",
            "evidence_sufficient",
            "privacy_scope_ok",
        )

        for j, v in enumerate(verdicts_raw):
            if not isinstance(v, dict):
                self._fail_bad_model_output(
                    context="verify_batch",
                    message=f"verdicts[{j}] must be an object, got {type(v).__name__}",
                    raw=raw,
                )

            for key in ("index", *required_bool_keys, "score", "reason"):
                if key not in v:
                    self._fail_bad_model_output(
                        context="verify_batch",
                        message=f"verdicts[{j}] missing required key: {key}",
                        raw=raw,
                    )

            idx_float = self._to_float(v.get("index"), default=float("nan"))
            if math.isnan(idx_float) or int(idx_float) != idx_float:
                self._fail_bad_model_output(
                    context="verify_batch",
                    message=f"verdicts[{j}].index must be an integer, got {v.get('index')!r}",
                    raw=raw,
                )
            idx = int(idx_float)
            if idx < 0 or idx >= len(verifier_items):
                self._fail_bad_model_output(
                    context="verify_batch",
                    message=f"verdicts[{j}].index out of range: {idx}; expected 0..{len(verifier_items) - 1}",
                    raw=raw,
                )
            if idx in by_idx:
                self._fail_bad_model_output(
                    context="verify_batch",
                    message=f"Duplicate verdict index: {idx}",
                    raw=raw,
                )

            for key in required_bool_keys:
                if not isinstance(v[key], bool):
                    self._fail_bad_model_output(
                        context="verify_batch",
                        message=f"verdicts[{j}].{key} must be bool, got {type(v[key]).__name__}",
                        raw=raw,
                    )

            score = self._to_float(v.get("score"), default=float("nan"))
            if math.isnan(score):
                self._fail_bad_model_output(
                    context="verify_batch",
                    message=f"verdicts[{j}].score must be numeric, got {v.get('score')!r}",
                    raw=raw,
                )

            by_idx[idx] = v

        missing = [i for i in range(len(verifier_items)) if i not in by_idx]
        if missing:
            self._fail_bad_model_output(
                context="verify_batch",
                message=f"Missing verdicts for candidate indexes: {missing}",
                raw=raw,
            )

        return [by_idx[i] for i in range(len(verifier_items))]

    # ---------------------------------------------------------------------
    # Ollama HTTP
    # ---------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        # bge-m3 handles multilingual, but very long text slows endpoint and can dilute vectors.
        text = self._head_tail(text, max_chars=3500)
        key = self._cache_key(self.embed_model, text)
        cached = FooSystem._embed_cache.get(key)
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

        FooSystem._embed_cache[key] = emb
        return emb

    def _chat_json(self, prompt: str, *, cache_prefix: str) -> str:
        key = self._cache_key(self.llm_model + ":" + cache_prefix, prompt)
        cached = FooSystem._chat_cache.get(key)
        if cached is not None:
            return cached

        payload = {
            "model": self.llm_model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON API. Output valid JSON only. Do not include analysis.",
                },
                {"role": "user", "content": prompt},
            ],
            "think": False,
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 4096,
            },
        }
        data = self._post_json("/api/chat", payload)
        content = str(data.get("message", {}).get("content", ""))
        FooSystem._chat_cache[key] = content
        return content

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
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"Ollama request failed: {url}: {last_err}")

    # ---------------------------------------------------------------------
    # Text handling
    # ---------------------------------------------------------------------

    @staticmethod
    def _memory_id(memory: dict[str, Any]) -> str:
        for key in ("memory_id", "id", "session_id"):
            if key in memory:
                return str(memory[key])
        raise KeyError(f"memory has no id field: {list(memory.keys())}")

    @staticmethod
    def _memory_text(memory: dict[str, Any]) -> str:
        # Do not use benchmark-only labels like is_target_memory/is_forbidden_memory.
        if "text" in memory:
            return str(memory["text"])
        turns = memory.get("turns")
        if isinstance(turns, list):
            parts = []
            for t in turns:
                if isinstance(t, dict):
                    parts.append(f"{t.get('role', '')}: {t.get('content', '')}")
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

    def _compact_memory_text(self, memory: dict[str, Any], *, max_chars: int) -> str:
        raw = self._memory_text(memory)
        key = self._cache_key("compact", raw + f"|{max_chars}")
        cached = FooSystem._compact_cache.get(key)
        if cached is not None:
            return cached

        snippets = self._self_statement_snippets(memory)
        if snippets:
            snippet_text = "\n".join(snippets[:28])
            compact = f"USER SELF-STATEMENT SNIPPETS:\n{snippet_text}\n\nRAW HEAD/TAIL:\n{self._head_tail(raw, max_chars=max_chars // 2)}"
        else:
            compact = self._head_tail(raw, max_chars=max_chars)
        compact = compact[:max_chars]
        FooSystem._compact_cache[key] = compact
        return compact

    @staticmethod
    def _self_statement_snippets(memory: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        turns = memory.get("turns")
        if isinstance(turns, list):
            for t in turns:
                if isinstance(t, dict) and str(t.get("role", "")).lower() == "user":
                    texts.append(str(t.get("content", "")))
        else:
            texts.append(FooSystem._memory_text(memory))

        first_person = re.compile(
            r"\b(i|i'm|i’ve|i've|i’d|i'd|i’ll|i'll|me|my|mine|we|our|ours)\b", re.I
        )
        signal = re.compile(
            r"\b(live|lived|work|worked|job|role|company|graduated|degree|major|school|study|studied|prefer|preference|favorite|like|own|have|has|use|using|account|email|phone|address|birthday|born|name|pet|dog|cat|car|device|laptop|computer|doctor|allergic|allergy|usually|always|commute|schedule|subscription|plan)\b",
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
        # stable de-dupe
        seen = set()
        uniq = []
        for s in out:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(s)
        return uniq

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
    def _simple_overlap(a: str, b: str) -> float:
        ta = set(FooSystem._tokenize(a))
        tb = set(FooSystem._tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, min(len(ta), len(tb)))

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
    def _parse_json_or_raise(raw: str, *, context: str, expected_schema: str) -> Any:
        if not raw or not raw.strip():
            FooSystem._fail_bad_model_output(
                context=context,
                message="Model returned empty output",
                raw=raw,
                expected_schema=expected_schema,
            )

        s = raw.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
            s = re.sub(r"```$", "", s).strip()

        try:
            return json.loads(s)
        except json.JSONDecodeError as first_error:
            try:
                repaired = repair_json(s)
                if not isinstance(repaired, str) or not repaired.strip():
                    FooSystem._fail_bad_model_output(
                        context=context,
                        message=(
                            f"JSON parse failed at line {first_error.lineno}, "
                            f"column {first_error.colno}: {first_error.msg}; "
                            "json-repair returned an empty repair"
                        ),
                        raw=raw,
                        expected_schema=expected_schema,
                    )

                obj = json.loads(repaired)
                FooSystem._print_json_repair_warning(
                    context=context,
                    first_error=first_error,
                    raw=raw,
                    repaired=repaired,
                )
                return obj
            except ModelOutputFormatError:
                raise
            except Exception as repair_error:
                FooSystem._fail_bad_model_output(
                    context=context,
                    message=(
                        f"JSON parse failed at line {first_error.lineno}, "
                        f"column {first_error.colno}: {first_error.msg}; "
                        f"json-repair also failed: {type(repair_error).__name__}: {repair_error}"
                    ),
                    raw=raw,
                    expected_schema=expected_schema,
                )

    @staticmethod
    def _print_json_repair_warning(
        *,
        context: str,
        first_error: json.JSONDecodeError,
        raw: str,
        repaired: str,
    ) -> None:
        print("\n" + "=" * 100, flush=True)
        print(f"FooSystem JSON repaired [{context}]", flush=True)
        print(
            f"Original parse error: line {first_error.lineno}, "
            f"column {first_error.colno}: {first_error.msg}",
            flush=True,
        )
        print("Raw model output START", flush=True)
        print(FooSystem._truncate_for_log(raw), flush=True)
        print("Raw model output END", flush=True)
        print("Repaired JSON START", flush=True)
        print(FooSystem._truncate_for_log(repaired), flush=True)
        print("Repaired JSON END", flush=True)
        print("=" * 100 + "\n", flush=True)

    @staticmethod
    def _truncate_for_log(text: str, max_chars: int = 6000) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n... [truncated for log] ...\n" + text[-half:]

    @staticmethod
    def _fail_bad_model_output(
        *,
        context: str,
        message: str,
        raw: str,
        expected_schema: str | None = None,
    ) -> None:
        print("\n" + "=" * 100, flush=True)
        print(f"FooSystem model output format error [{context}]", flush=True)
        print(f"Error: {message}", flush=True)
        if expected_schema is not None:
            print("Expected schema:", flush=True)
            print(expected_schema, flush=True)
        print("Raw model output START", flush=True)
        print(raw, flush=True)
        print("Raw model output END", flush=True)
        print("=" * 100 + "\n", flush=True)
        raise ModelOutputFormatError(f"{context}: {message}")

    @staticmethod
    def _safe_json(text: str, *, default: Any) -> Any:
        # Kept only for backwards compatibility with older local experiments.
        # New FooSystem paths use _parse_json_or_raise(...) and fail fast.
        if not text:
            return default
        s = text.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
            s = re.sub(r"```$", "", s).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return default

    @staticmethod
    def _to_float(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _cache_key(prefix: str, text: str) -> str:
        return prefix + ":" + FooSystem._hash(text)

    @staticmethod
    def _brief(obj: Any, max_chars: int = 450) -> str:
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return s if len(s) <= max_chars else s[:max_chars] + "..."
