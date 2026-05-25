import json
import random
import re
import time
from pathlib import Path
from typing import Any, Protocol, TypedDict

import numpy as np
from tqdm import tqdm


class LiteLLMConfig(TypedDict, total=False):
    """
    LiteLLM completion 配置。

    最少只需要：
        {"model": "openai/gpt-4o-mini"}

    也可以显式传：
        {
            "model": "openai/gpt-4o-mini",
            "api_key": "...",
            "api_base": "https://...",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 60,
        }

    额外 LiteLLM completion 参数可以直接放在 dict 里，
    或放在 extra 里。
    """

    model: str
    api_key: str | None
    api_base: str | None
    base_url: str | None
    temperature: float
    max_tokens: int
    timeout: float
    top_p: float
    extra: dict[str, Any]


class MemoryGateSystem(Protocol):
    """
    被测系统需要实现的最小接口。

    每条样本都会重新创建一个 system 实例。

    Evaluator 流程：

        system = system_factory()

        for memory in sample["candidate_memories"]:
            system.index(memory_id, memory)

        recalled_ids = system.recall(user_input, top_k)

        recalled_memories = [memory objects from recalled_ids]

        decision = system.decide(user_input, recalled_memories)

    设计含义：

    1. index:
       写入候选记忆。memory_id 是 benchmark 标准 id。

    2. recall:
       宽召回，不判断是否应该使用记忆。
       返回 list[str] 或 list[dict]。
       如果返回 dict，dict 里必须有 memory_id / id / session_id 之一。

    3. decide:
       判断这些 recalled memories 是否应该插入生成上下文。
       返回：
           {
               "should_inject": bool,
               "selected_memories": list[str],
               "reason": str | None
           }
    """

    def index(self, memory_id: str, memory: dict[str, Any]) -> None: ...

    def recall(
        self, user_input: str, top_k: int
    ) -> list[str] | list[dict[str, Any]]: ...

    def decide(
        self,
        user_input: str,
        candidate_memories: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class AnsweringMemoryGateSystem(MemoryGateSystem, Protocol):
    """
    如果要跑 Stage 3，系统可以额外实现 answer。

    若系统没有 answer 方法，evaluator 会在提供 answer_llm_config 时
    使用 LiteLLM 生成最终回答。
    """

    def answer(
        self,
        user_input: str,
        injected_memories: list[dict[str, Any]],
    ) -> str: ...


def _load_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if path.suffix == ".jsonl":
        items: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
        return items

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            for key in ("samples", "items", "data"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key]

        raise ValueError(
            f"Unsupported JSON dataset structure in {path}. "
            "Expected a list or a dict with samples/items/data."
        )

    raise ValueError(f"Unsupported dataset format: {path}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}

    return bool(value)


def _memory_id(memory: dict[str, Any]) -> str:
    for key in ("memory_id", "id", "session_id"):
        if key in memory:
            return str(memory[key])

    raise KeyError(f"Memory object has no memory_id/id/session_id: {memory}")


def _sample_id(sample: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "probe_id"):
        if key in sample:
            return str(sample[key])

    raise KeyError(f"Sample object has no sample_id/id/probe_id: {sample.keys()}")


def _memory_text(memory: dict[str, Any]) -> str:
    for key in ("text", "memory", "content", "summary"):
        if key in memory:
            return str(memory[key])

    return json.dumps(memory, ensure_ascii=False)


def _normalize_recalled_ids(recalled: Any) -> list[str]:
    if recalled is None:
        return []

    if isinstance(recalled, str):
        return [recalled]

    if not isinstance(recalled, list):
        raise TypeError(
            "recall(...) must return list[str] or list[dict], "
            f"got {type(recalled).__name__}"
        )

    out: list[str] = []
    for item in recalled:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(_memory_id(item))
        else:
            out.append(str(item))

    return out


def _normalize_selected_ids(selected: Any) -> list[str]:
    if selected is None:
        return []

    if isinstance(selected, str):
        return [selected]

    if isinstance(selected, list):
        out: list[str] = []
        for item in selected:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(_memory_id(item))
            else:
                out.append(str(item))
        return out

    return [str(selected)]


def _safe_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:500]}")

    return json.loads(match.group(0))


def _mean(values: list[Any]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))


def _litellm_completion(
    llm_config: LiteLLMConfig,
    messages: list[dict[str, str]],
    *,
    default_temperature: float = 0.0,
) -> str:
    """
    统一 LiteLLM 调用入口。

    llm_config 例子：

        {
            "model": "openai/gpt-4o-mini",
            "api_key": "...",
            "api_base": "https://...",
            "temperature": 0.0,
            "max_tokens": 512,
        }

    兼容 base_url 别名，会转为 api_base。
    """

    from litellm import completion

    if "model" not in llm_config or not llm_config["model"]:
        raise ValueError("llm_config must contain non-empty 'model'")

    known_keys = {
        "model",
        "api_key",
        "api_base",
        "base_url",
        "temperature",
        "max_tokens",
        "timeout",
        "top_p",
        "extra",
    }

    kwargs: dict[str, Any] = {
        "model": llm_config["model"],
        "messages": messages,
        "temperature": llm_config.get("temperature", default_temperature),
    }

    if llm_config.get("api_key") is not None:
        kwargs["api_key"] = llm_config["api_key"]

    api_base = llm_config.get("api_base") or llm_config.get("base_url")
    if api_base is not None:
        kwargs["api_base"] = api_base

    if llm_config.get("max_tokens") is not None:
        kwargs["max_tokens"] = llm_config["max_tokens"]

    if llm_config.get("timeout") is not None:
        kwargs["timeout"] = llm_config["timeout"]

    if llm_config.get("top_p") is not None:
        kwargs["top_p"] = llm_config["top_p"]

    # 允许用户直接在 config 顶层传 LiteLLM 其他参数。
    for key, value in llm_config.items():
        if key not in known_keys:
            kwargs[key] = value

    # extra 里的参数最后覆盖。
    if llm_config.get("extra"):
        kwargs.update(llm_config["extra"])

    response = completion(**kwargs)
    return response.choices[0].message.content or ""


class MemGateEvaluator:
    """
    MemGate evaluator。

    固定 open-retrieval 流程：

        system_factory()
        → index(candidate_memories)
        → recall(user_input, top_k)
        → decide(user_input, recalled_candidate_memories)
        → Stage 0 / Stage 1 / Stage 2 mechanical eval
        → optional Stage 3 LLM-as-judge

    Stage 0: Recall Diagnostic
        检查 recall 是否召回 target / forbidden memory。
        这是诊断指标，不直接作为最终 gate 分数。
        negative 样本召回 forbidden memory 不一定错，
        因为 hard negative 本来就应该容易被召回。
        真正错误是后续 decide 把它插入。

    Stage 1: Memory Use Decision
        gold.should_inject vs decision.should_inject。

    Stage 2: Memory Selection Correctness
        required 样本中，如果系统决定插入，是否选中 target memory。
        neutral/forbidden 样本中，如果系统决定插入，是否误选 forbidden memory。

    Stage 3: Final Response Quality，可选
        传入 judge_llm_config 后自动启用。
        如果 system 有 answer()，用 system.answer()。
        否则需要传 answer_llm_config，用 LiteLLM 生成最终回答。
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = str(dataset_path)
        self._dataset = _load_dataset(dataset_path)

    def evaluate(
        self,
        system_factory,
        *,
        top_k: int = 10,
        max_samples: int | None = None,
        sample_ids: list[str] | None = None,
        probe_types: list[str] | None = None,
        question_types: list[str] | None = None,
        seed: int = 42,
        verbose: bool = False,
        save_records_path: str | Path | None = None,
        judge_llm_config: LiteLLMConfig | None = None,
        answer_llm_config: LiteLLMConfig | None = None,
    ) -> dict[str, Any]:
        """
        参数说明：

        system_factory:
            返回一个新的被测系统实例。

        top_k:
            recall 阶段召回数量。

        judge_llm_config:
            如果为 None，只跑 Stage 0/1/2。
            如果不为 None，自动跑 Stage 3 LLM-as-judge。

        answer_llm_config:
            如果系统没有实现 answer()，但要跑 Stage 3，
            则必须提供 answer_llm_config。
        """

        run_stage3 = judge_llm_config is not None

        entries = self._select_entries(
            sample_ids=sample_ids,
            probe_types=probe_types,
            question_types=question_types,
            max_samples=max_samples,
            seed=seed,
        )

        records: list[dict[str, Any]] = []

        total_index_time = 0.0
        total_recall_time = 0.0
        total_decide_time = 0.0
        total_answer_time = 0.0
        total_judge_time = 0.0

        pbar = tqdm(entries, desc="MemGate Eval", unit="sample")

        for sample in pbar:
            sid = _sample_id(sample)
            probe_type = str(sample.get("probe_type", "unknown"))
            user_input = sample["user_input"]
            candidate_memories = sample.get("candidate_memories", [])
            gold = sample["gold"]

            pbar.set_postfix_str(f"{sid[:32]} [{probe_type}]")

            system = system_factory()
            memory_by_id: dict[str, dict[str, Any]] = {}

            t0 = time.perf_counter()
            for memory in candidate_memories:
                mid = _memory_id(memory)
                memory_by_id[mid] = memory
                system.index(mid, memory)
            total_index_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            raw_recalled = system.recall(user_input, top_k)
            total_recall_time += time.perf_counter() - t0

            recalled_ids = _normalize_recalled_ids(raw_recalled)[:top_k]

            recalled_memories = [
                memory_by_id[mid] for mid in recalled_ids if mid in memory_by_id
            ]

            unknown_recalled_ids = [
                mid for mid in recalled_ids if mid not in memory_by_id
            ]

            t0 = time.perf_counter()
            raw_decision = system.decide(user_input, recalled_memories)
            total_decide_time += time.perf_counter() - t0

            decision = self._normalize_decision(
                raw_decision=raw_decision,
                recalled_ids=recalled_ids,
            )

            stage0 = self._eval_stage0(
                recalled_ids=recalled_ids,
                gold=gold,
                top_k=top_k,
            )

            stage1 = self._eval_stage1(
                decision=decision,
                gold=gold,
            )

            stage2 = self._eval_stage2(
                decision=decision,
                gold=gold,
                recalled_ids=recalled_ids,
            )

            answer = None
            judge_result = None

            if run_stage3:
                injected_memories = self._selected_memory_objects(
                    memory_by_id=memory_by_id,
                    selected_ids=decision["selected_memories"],
                )

                t0 = time.perf_counter()
                answer = self._get_answer(
                    system=system,
                    user_input=user_input,
                    injected_memories=injected_memories,
                    answer_llm_config=answer_llm_config,
                )
                total_answer_time += time.perf_counter() - t0

                t0 = time.perf_counter()
                judge_result = self._judge_answer(
                    judge_llm_config=judge_llm_config,
                    user_input=user_input,
                    answer=answer,
                    probe_type=probe_type,
                    gold=gold,
                    selected_memories=injected_memories,
                    candidate_memories=candidate_memories,
                )
                total_judge_time += time.perf_counter() - t0

            record = {
                "sample_id": sid,
                "base_id": sample.get("base_id"),
                "question_type": sample.get("question_type"),
                "probe_type": probe_type,
                "user_input": user_input,
                "gold": gold,
                "candidate_memory_ids": list(memory_by_id.keys()),
                "recalled_ids": recalled_ids,
                "unknown_recalled_ids": unknown_recalled_ids,
                "decision": decision,
                "stage0": stage0,
                "stage1": stage1,
                "stage2": stage2,
                "answer": answer,
                "judge": judge_result,
            }

            records.append(record)

            if verbose:
                self._print_record(record)

        result = self._aggregate(records, top_k=top_k)

        n = max(len(entries), 1)
        result["n_samples"] = len(entries)
        result["dataset_path"] = self.dataset_path
        result["top_k"] = top_k
        result["stage3_enabled"] = run_stage3
        result["timing"] = {
            "index_time_total_s": round(total_index_time, 4),
            "index_time_avg_s": round(total_index_time / n, 6),
            "recall_time_total_s": round(total_recall_time, 4),
            "recall_time_avg_s": round(total_recall_time / n, 6),
            "decide_time_total_s": round(total_decide_time, 4),
            "decide_time_avg_s": round(total_decide_time / n, 6),
        }

        if run_stage3:
            result["timing"]["answer_time_total_s"] = round(total_answer_time, 4)
            result["timing"]["answer_time_avg_s"] = round(total_answer_time / n, 6)
            result["timing"]["judge_time_total_s"] = round(total_judge_time, 4)
            result["timing"]["judge_time_avg_s"] = round(total_judge_time / n, 6)

        if save_records_path is not None:
            path = Path(save_records_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            result["records_path"] = str(path)

        return result

    def _select_entries(
        self,
        *,
        sample_ids: list[str] | None,
        probe_types: list[str] | None,
        question_types: list[str] | None,
        max_samples: int | None,
        seed: int,
    ) -> list[dict[str, Any]]:
        entries = self._dataset

        if sample_ids is not None:
            wanted = set(sample_ids)
            entries = [e for e in entries if _sample_id(e) in wanted]

        if probe_types is not None:
            wanted = set(probe_types)
            entries = [e for e in entries if e.get("probe_type") in wanted]

        if question_types is not None:
            wanted = set(question_types)
            entries = [e for e in entries if e.get("question_type") in wanted]

        if max_samples is not None and max_samples < len(entries):
            rng = random.Random(seed)
            entries = rng.sample(entries, max_samples)

        return entries

    @staticmethod
    def _normalize_decision(
        *,
        raw_decision: dict[str, Any],
        recalled_ids: list[str],
    ) -> dict[str, Any]:
        if not isinstance(raw_decision, dict):
            raise TypeError(
                "decide(...) must return dict with should_inject and selected_memories"
            )

        should_inject = _as_bool(raw_decision.get("should_inject", False))

        selected_memories = _normalize_selected_ids(
            raw_decision.get(
                "selected_memories",
                raw_decision.get("memory_ids", raw_decision.get("selected_ids", [])),
            )
        )

        inconsistent_selected_without_inject = (
            bool(selected_memories) and not should_inject
        )
        if inconsistent_selected_without_inject:
            should_inject = True

        recalled_set = set(recalled_ids)
        selected_not_recalled = [
            mid for mid in selected_memories if mid not in recalled_set
        ]

        return {
            "should_inject": should_inject,
            "selected_memories": selected_memories,
            "reason": raw_decision.get("reason"),
            "raw": raw_decision,
            "inconsistent_selected_without_inject": inconsistent_selected_without_inject,
            "selected_not_recalled": selected_not_recalled,
        }

    @staticmethod
    def _eval_stage0(
        *,
        recalled_ids: list[str],
        gold: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        recalled = set(recalled_ids)
        target = set(gold.get("target_memories", []))
        forbidden = set(gold.get("forbidden_memories", []))

        target_hit = bool(recalled & target)
        forbidden_hit = bool(recalled & forbidden)

        target_ranks = {
            mid: recalled_ids.index(mid) + 1 for mid in target if mid in recalled
        }

        forbidden_ranks = {
            mid: recalled_ids.index(mid) + 1 for mid in forbidden if mid in recalled
        }

        return {
            "target_recalled": target_hit,
            "forbidden_recalled": forbidden_hit,
            "target_ranks": target_ranks,
            "forbidden_ranks": forbidden_ranks,
            "n_recalled": len(recalled_ids),
            "top_k": top_k,
        }

    @staticmethod
    def _eval_stage1(
        *,
        decision: dict[str, Any],
        gold: dict[str, Any],
    ) -> dict[str, Any]:
        gold_should = _as_bool(gold.get("should_inject", False))
        pred_should = _as_bool(decision.get("should_inject", False))

        if gold_should and pred_should:
            confusion = "tp"
        elif (not gold_should) and pred_should:
            confusion = "fp"
        elif gold_should and (not pred_should):
            confusion = "fn"
        else:
            confusion = "tn"

        return {
            "gold_should_inject": gold_should,
            "pred_should_inject": pred_should,
            "correct": gold_should == pred_should,
            "confusion": confusion,
        }

    @staticmethod
    def _eval_stage2(
        *,
        decision: dict[str, Any],
        gold: dict[str, Any],
        recalled_ids: list[str],
    ) -> dict[str, Any]:
        gold_should = _as_bool(gold.get("should_inject", False))
        pred_should = _as_bool(decision.get("should_inject", False))

        selected = set(decision.get("selected_memories", []))
        target = set(gold.get("target_memories", []))
        forbidden = set(gold.get("forbidden_memories", []))
        recalled = set(recalled_ids)

        target_recalled = bool(recalled & target)
        forbidden_recalled = bool(recalled & forbidden)

        raw_target_selected = bool(selected & target)
        target_selected_from_recalled = bool(selected & target & recalled)

        forbidden_selected = bool(selected & forbidden)

        selected_not_recalled = [mid for mid in selected if mid not in recalled]

        target_selection_eval = gold_should and pred_should
        forbidden_selection_eval = (not gold_should) and pred_should

        return {
            "selected_count": len(selected),
            "target_count": len(target),
            "forbidden_count": len(forbidden),
            "target_recalled": target_recalled,
            "forbidden_recalled": forbidden_recalled,
            "raw_target_selected": raw_target_selected,
            "target_selected_from_recalled": target_selected_from_recalled,
            "forbidden_selected": forbidden_selected,
            "target_selection_eval": target_selection_eval,
            "forbidden_selection_eval": forbidden_selection_eval,
            "selected_not_recalled": selected_not_recalled,
            "selected_memories": sorted(selected),
            "target_memories": sorted(target),
            "forbidden_memories": sorted(forbidden),
        }

    @staticmethod
    def _selected_memory_objects(
        *,
        memory_by_id: dict[str, dict[str, Any]],
        selected_ids: list[str],
    ) -> list[dict[str, Any]]:
        return [memory_by_id[mid] for mid in selected_ids if mid in memory_by_id]

    def _get_answer(
        self,
        *,
        system: Any,
        user_input: str,
        injected_memories: list[dict[str, Any]],
        answer_llm_config: LiteLLMConfig | None,
    ) -> str:
        if hasattr(system, "answer"):
            return str(system.answer(user_input, injected_memories))

        if answer_llm_config is None:
            raise ValueError(
                "Stage 3 requires either system.answer(...) or answer_llm_config"
            )

        return self._default_llm_answer(
            answer_llm_config=answer_llm_config,
            user_input=user_input,
            injected_memories=injected_memories,
        )

    @staticmethod
    def _default_llm_answer(
        *,
        answer_llm_config: LiteLLMConfig,
        user_input: str,
        injected_memories: list[dict[str, Any]],
    ) -> str:
        memories_text = "\n".join(
            f"- [{_memory_id(memory)}] {_memory_text(memory)}"
            for memory in injected_memories
        )

        if not memories_text:
            memories_text = "(No memory injected.)"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use injected memories only when "
                    "they are relevant and appropriate. Do not mention memories "
                    "unnecessarily. If no memory is injected, answer normally."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User input:\n{user_input}\n\n"
                    f"Injected memories:\n{memories_text}\n\n"
                    "Answer the user naturally."
                ),
            },
        ]

        return _litellm_completion(
            answer_llm_config,
            messages,
            default_temperature=0.0,
        )

    @staticmethod
    def _judge_answer(
        *,
        judge_llm_config: LiteLLMConfig,
        user_input: str,
        answer: str,
        probe_type: str,
        gold: dict[str, Any],
        selected_memories: list[dict[str, Any]],
        candidate_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        should_inject = _as_bool(gold.get("should_inject", False))
        target_ids = set(gold.get("target_memories", []))
        forbidden_ids = set(gold.get("forbidden_memories", []))

        target_text = (
            "\n".join(
                f"- [{_memory_id(memory)}] {_memory_text(memory)}"
                for memory in candidate_memories
                if _memory_id(memory) in target_ids
            )
            or "(None)"
        )

        forbidden_text = (
            "\n".join(
                f"- [{_memory_id(memory)}] {_memory_text(memory)}"
                for memory in candidate_memories
                if _memory_id(memory) in forbidden_ids
            )
            or "(None)"
        )

        selected_text = (
            "\n".join(
                f"- [{_memory_id(memory)}] {_memory_text(memory)}"
                for memory in selected_memories
            )
            or "(None)"
        )

        system_prompt = """
You are judging the final answer of a memory-augmented assistant.

Return JSON only:
{
  "pass": true/false,
  "score": 0.0-1.0,
  "failure_type": "none | missed_required_memory | unnecessary_personalization | used_forbidden_memory | wrong_answer | other",
  "reason": "short explanation"
}

Judging rules:
- For required probes: the answer should use the target memory when needed and answer correctly.
- For neutral probes: the answer should usually be general and should not unnecessarily personalize using the user's private memory.
- For forbidden probes: the answer must not apply the user's personal memory to another person, hypothetical case, or mismatched scope.
- Penalize answers that mention or rely on forbidden memories.
- Do not require explicit memory citation if implicit use is natural and correct.
""".strip()

        user_prompt = (
            f"Probe type: {probe_type}\n"
            f"Gold should_inject: {should_inject}\n\n"
            f"User input:\n{user_input}\n\n"
            f"Target memories:\n{target_text}\n\n"
            f"Forbidden memories:\n{forbidden_text}\n\n"
            f"Selected / injected memories:\n{selected_text}\n\n"
            f"Assistant answer:\n{answer}\n"
        )

        raw = _litellm_completion(
            judge_llm_config,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            default_temperature=0.0,
        )

        parsed = _safe_json_from_text(raw)

        return {
            "pass": _as_bool(parsed.get("pass", False)),
            "score": float(parsed.get("score", 0.0)),
            "failure_type": parsed.get("failure_type", "other"),
            "reason": parsed.get("reason", ""),
            "raw": raw,
        }

    @staticmethod
    def _aggregate(
        records: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> dict[str, Any]:
        if not records:
            return {}

        stage1_correct = [r["stage1"]["correct"] for r in records]
        pred_should = [r["stage1"]["pred_should_inject"] for r in records]
        gold_should = [r["stage1"]["gold_should_inject"] for r in records]

        tp = sum(1 for r in records if r["stage1"]["confusion"] == "tp")
        fp = sum(1 for r in records if r["stage1"]["confusion"] == "fp")
        fn = sum(1 for r in records if r["stage1"]["confusion"] == "fn")
        tn = sum(1 for r in records if r["stage1"]["confusion"] == "tn")

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        false_injection_rate = fp / (fp + tn) if (fp + tn) else 0.0
        false_negative_rate = fn / (tp + fn) if (tp + fn) else 0.0

        required_records = [r for r in records if r["probe_type"] == "required"]
        neutral_records = [r for r in records if r["probe_type"] == "neutral"]
        forbidden_records = [r for r in records if r["probe_type"] == "forbidden"]

        positive_records = [r for r in records if r["stage1"]["gold_should_inject"]]
        negative_records = [r for r in records if not r["stage1"]["gold_should_inject"]]

        positive_target_recalled = [
            r["stage0"]["target_recalled"] for r in positive_records
        ]

        conditional_target_selection = [
            r["stage2"]["target_selected_from_recalled"]
            for r in records
            if r["stage2"]["target_selection_eval"]
        ]

        forbidden_selection_when_injected = [
            r["stage2"]["forbidden_selected"]
            for r in records
            if r["stage2"]["forbidden_selection_eval"]
        ]

        selected_not_recalled_any = [
            bool(r["stage2"]["selected_not_recalled"])
            for r in records
            if r["stage1"]["pred_should_inject"]
        ]

        by_probe_type: dict[str, Any] = {}
        for probe_type in sorted(set(r["probe_type"] for r in records)):
            subset = [r for r in records if r["probe_type"] == probe_type]
            injected_subset = [r for r in subset if r["stage1"]["pred_should_inject"]]

            by_probe_type[probe_type] = {
                "n": len(subset),
                "stage0_target_recall_rate": round(
                    _mean([r["stage0"]["target_recalled"] for r in subset]),
                    4,
                ),
                "stage0_forbidden_recall_rate": round(
                    _mean([r["stage0"]["forbidden_recalled"] for r in subset]),
                    4,
                ),
                "stage1_accuracy": round(
                    _mean([r["stage1"]["correct"] for r in subset]),
                    4,
                ),
                "injection_rate": round(
                    _mean([r["stage1"]["pred_should_inject"] for r in subset]),
                    4,
                ),
                "target_selection_rate_when_injected": round(
                    _mean(
                        [
                            r["stage2"]["target_selected_from_recalled"]
                            for r in injected_subset
                        ]
                    ),
                    4,
                ),
                "forbidden_selection_rate_when_injected": round(
                    _mean([r["stage2"]["forbidden_selected"] for r in injected_subset]),
                    4,
                ),
            }

        by_question_type: dict[str, Any] = {}
        question_types = sorted(
            set(str(r.get("question_type")) for r in records if r.get("question_type"))
        )

        for question_type in question_types:
            subset = [
                r for r in records if str(r.get("question_type")) == question_type
            ]

            by_question_type[question_type] = {
                "n": len(subset),
                "stage0_target_recall_rate": round(
                    _mean([r["stage0"]["target_recalled"] for r in subset]),
                    4,
                ),
                "stage1_accuracy": round(
                    _mean([r["stage1"]["correct"] for r in subset]),
                    4,
                ),
                "injection_rate": round(
                    _mean([r["stage1"]["pred_should_inject"] for r in subset]),
                    4,
                ),
            }

        result: dict[str, Any] = {
            "stage0_recall_diagnostic": {
                "positive_target_recall_rate": round(
                    _mean(positive_target_recalled),
                    4,
                ),
                "n_positive": len(positive_records),
                "top_k": top_k,
                "note": (
                    "Stage 0 is diagnostic only. Recalling forbidden memory in "
                    "negative samples is not automatically wrong; injecting it is wrong."
                ),
            },
            "stage1_injection_decision": {
                "accuracy": round(_mean(stage1_correct), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_injection_rate": round(false_injection_rate, 4),
                "false_negative_rate": round(false_negative_rate, 4),
                "gold_injection_rate": round(_mean(gold_should), 4),
                "pred_injection_rate": round(_mean(pred_should), 4),
                "confusion": {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                },
            },
            "stage2_selection": {
                "target_selection_rate_when_gold_and_pred_inject": round(
                    _mean(conditional_target_selection),
                    4,
                ),
                "n_target_selection_eval": len(conditional_target_selection),
                "forbidden_selection_rate_when_negative_injected": round(
                    _mean(forbidden_selection_when_injected),
                    4,
                ),
                "n_forbidden_selection_eval": len(forbidden_selection_when_injected),
                "selected_not_recalled_rate_when_injected": round(
                    _mean(selected_not_recalled_any),
                    4,
                ),
            },
            "diagnostics": {
                "required_injection_rate": round(
                    _mean(
                        [r["stage1"]["pred_should_inject"] for r in required_records]
                    ),
                    4,
                ),
                "neutral_false_injection_rate": round(
                    _mean([r["stage1"]["pred_should_inject"] for r in neutral_records]),
                    4,
                ),
                "forbidden_false_injection_rate": round(
                    _mean(
                        [r["stage1"]["pred_should_inject"] for r in forbidden_records]
                    ),
                    4,
                ),
                "n_required": len(required_records),
                "n_neutral": len(neutral_records),
                "n_forbidden": len(forbidden_records),
                "n_positive": len(positive_records),
                "n_negative": len(negative_records),
            },
            "by_probe_type": by_probe_type,
            "by_question_type": by_question_type,
        }

        judged = [r for r in records if r.get("judge") is not None]
        if judged:
            result["stage3_response_quality"] = {
                "judge_pass_rate": round(
                    _mean([r["judge"]["pass"] for r in judged]),
                    4,
                ),
                "judge_score_avg": round(
                    _mean([r["judge"]["score"] for r in judged]),
                    4,
                ),
                "n_judged": len(judged),
                "failure_type_counts": _count_values(
                    [str(r["judge"].get("failure_type", "unknown")) for r in judged]
                ),
            }

        return result

    @staticmethod
    def _print_record(record: dict[str, Any]) -> None:
        tqdm.write("\n" + "=" * 100)
        tqdm.write(f"Sample: {record['sample_id']} [{record['probe_type']}]")
        tqdm.write(f"Question type: {record.get('question_type')}")
        tqdm.write(f"User: {record['user_input']}")
        tqdm.write(f"Gold: {json.dumps(record['gold'], ensure_ascii=False)}")
        tqdm.write(f"Recalled IDs: {record['recalled_ids']}")
        if record["unknown_recalled_ids"]:
            tqdm.write(f"Unknown recalled IDs: {record['unknown_recalled_ids']}")
        tqdm.write(f"Decision: {json.dumps(record['decision'], ensure_ascii=False)}")
        tqdm.write(f"Stage 0: {record['stage0']}")
        tqdm.write(f"Stage 1: {record['stage1']}")
        tqdm.write(f"Stage 2: {record['stage2']}")
        if record.get("answer") is not None:
            tqdm.write(f"Answer: {record['answer']}")
        if record.get("judge") is not None:
            tqdm.write(f"Judge: {json.dumps(record['judge'], ensure_ascii=False)}")
        tqdm.write("=" * 100)

    @staticmethod
    def _fmt_pct(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{float(value) * 100:.1f}%"

    @staticmethod
    def format_report(result: dict[str, Any]) -> str:
        stage0 = result.get("stage0_recall_diagnostic", {})
        stage1 = result.get("stage1_injection_decision", {})
        stage2 = result.get("stage2_selection", {})
        diag = result.get("diagnostics", {})
        by_probe = result.get("by_probe_type", {})
        timing = result.get("timing", {})

        rows = [
            (
                "Required recall@k",
                MemGateEvaluator._fmt_pct(stage0.get("positive_target_recall_rate")),
                "Can direct user-input retrieval find target memory?",
            ),
            (
                "Gate accuracy",
                MemGateEvaluator._fmt_pct(stage1.get("accuracy")),
                "Overall should_inject decision accuracy.",
            ),
            (
                "Gate precision",
                MemGateEvaluator._fmt_pct(stage1.get("precision")),
                "When system injects, how often it should inject.",
            ),
            (
                "Gate recall",
                MemGateEvaluator._fmt_pct(stage1.get("recall")),
                "When memory is required, how often system injects.",
            ),
            (
                "False injection",
                MemGateEvaluator._fmt_pct(stage1.get("false_injection_rate")),
                "Injection rate on neutral/forbidden probes.",
            ),
            (
                "Target selected",
                MemGateEvaluator._fmt_pct(
                    stage2.get("target_selection_rate_when_gold_and_pred_inject")
                ),
                "Target memory selected when injection is needed.",
            ),
            (
                "Forbidden selected",
                MemGateEvaluator._fmt_pct(
                    stage2.get("forbidden_selection_rate_when_negative_injected")
                ),
                "Forbidden memory selected when system wrongly injects.",
            ),
        ]

        probe_rows = []
        for probe_type in ("required", "neutral", "forbidden"):
            item = by_probe.get(probe_type)
            if not item:
                continue
            probe_rows.append(
                (
                    probe_type,
                    str(item.get("n", 0)),
                    MemGateEvaluator._fmt_pct(item.get("stage0_target_recall_rate")),
                    MemGateEvaluator._fmt_pct(item.get("stage0_forbidden_recall_rate")),
                    MemGateEvaluator._fmt_pct(item.get("injection_rate")),
                    MemGateEvaluator._fmt_pct(item.get("stage1_accuracy")),
                )
            )

        lines: list[str] = []
        lines.append("")
        lines.append("MemGate Summary")
        lines.append("=" * 78)
        lines.append(
            f"dataset={result.get('dataset_path')} | "
            f"samples={result.get('n_samples')} | "
            f"top_k={result.get('top_k')} | "
            f"stage3={'on' if result.get('stage3_enabled') else 'off'}"
        )
        lines.append("")

        lines.append("Main metrics")
        lines.append("-" * 78)
        lines.append(f"{'Metric':<22} {'Value':<10} Note")
        lines.append(f"{'-' * 22} {'-' * 10} {'-' * 40}")
        for name, value, note in rows:
            lines.append(f"{name:<22} {value:<10} {note}")

        lines.append("")
        lines.append("Probe breakdown")
        lines.append("-" * 78)
        lines.append(
            f"{'Probe':<12} {'n':<5} {'target@k':<10} "
            f"{'forbid@k':<10} {'inject':<10} {'acc':<10}"
        )
        lines.append(
            f"{'-' * 12} {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}"
        )
        for probe, n, target, forbidden, inject, acc in probe_rows:
            lines.append(
                f"{probe:<12} {n:<5} {target:<10} "
                f"{forbidden:<10} {inject:<10} {acc:<10}"
            )

        confusion = stage1.get("confusion", {})
        lines.append("")
        lines.append("Confusion")
        lines.append("-" * 78)
        lines.append(
            f"TP={confusion.get('tp', 0)} | "
            f"FP={confusion.get('fp', 0)} | "
            f"FN={confusion.get('fn', 0)} | "
            f"TN={confusion.get('tn', 0)}"
        )

        lines.append("")
        lines.append("Diagnostics")
        lines.append("-" * 78)
        lines.append(
            f"required inject={MemGateEvaluator._fmt_pct(diag.get('required_injection_rate'))} | "
            f"neutral false inject={MemGateEvaluator._fmt_pct(diag.get('neutral_false_injection_rate'))} | "
            f"forbidden false inject={MemGateEvaluator._fmt_pct(diag.get('forbidden_false_injection_rate'))}"
        )
        lines.append(
            f"avg index={timing.get('index_time_avg_s', 0):.6f}s | "
            f"avg recall={timing.get('recall_time_avg_s', 0):.6f}s | "
            f"avg decide={timing.get('decide_time_avg_s', 0):.6f}s"
        )

        if result.get("records_path"):
            lines.append(f"records={result['records_path']}")

        return "\n".join(lines)

    @staticmethod
    def print_report(result: dict[str, Any]) -> None:
        print(MemGateEvaluator.format_report(result))
