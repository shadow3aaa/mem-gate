# MemGate Evaluation Rules

MemGate is intended to evaluate **general proactive memory retrieval and memory-use gating**, not handcrafted solvers tuned to a particular benchmark release.

A valid system must be able to run unchanged on non-MemGate user-memory data. Systems that exploit MemGate-specific prompt families, field vocabularies, fixed name lists, or test-set artifacts are considered **artifact-exploit baselines**, not compliant benchmark submissions.

## 1. Evaluation Goal

MemGate evaluates an end-to-end memory pipeline:

```text
user_input → recall candidate memories → decide whether to inject memory → select memories
```

The benchmark is designed to measure two capabilities:

1. **Query-gap retrieval**: required probes may need user memory even when the user input does not directly resemble the original LongMemEval query.
2. **Memory-use gating**: neutral and forbidden probes may retrieve tempting but inappropriate private memories; a good system should avoid injecting them.

MemGate is **not** a prompt-template recognition task. A system should make decisions from generalizable reasoning over the user input and memory contents.

## 2. Runtime Inputs

At runtime, a compliant system may use only:

- `user_input`
- textual content of indexed memories
- `memory_id` only as an opaque handle for indexing and returning selected memories

A compliant system must ignore all benchmark metadata, including but not limited to:

- `sample_id`
- `base_id`
- `probe_id`
- `probe_type`
- `question_type`
- `gold`
- `diagnostics`
- `applicability`
- `source_question`
- `source_answer`
- `target_memories`
- `forbidden_memories`
- `is_target_memory`
- any other field that directly or indirectly reveals benchmark construction or labels

If these fields are present in a memory object or sample object, they must not be used for retrieval, gating, ranking, feature extraction, prompting, or decision making.

## 3. Allowed Methods

The following are allowed when implemented as general-purpose memory-system techniques:

### 3.1 Retrieval

Allowed retrieval methods include:

- BM25 or other lexical retrieval
- embedding retrieval
- hybrid retrieval
- reranking
- multi-query retrieval
- LLM-generated query rewriting
- generic entity, temporal, or semantic expansion

A query constructor is allowed if it is not hand-coded against MemGate-specific fields, prompt families, or test-set vocabulary.

### 3.2 Memory-use gating

Allowed gating methods include:

- LLM classifiers
- learned classifiers trained only on the allowed training/dev data
- general privacy/scope rules
- generic coreference, ownership, and subject-scope reasoning
- rules that apply broadly to arbitrary user-memory systems

Examples of allowed general principles:

```text
Do not apply the user's private memory to a third party.
Do not inject personal memory into a generic knowledge question.
Inject memory when the user is asking for a value that depends on their own prior context.
Avoid exposing private memory unless it is necessary for the current request.
```

### 3.3 Memory content features

General memory-content features are allowed, such as:

- recency
- conversation length
- entity density
- numerical specificity
- first-person user statements
- whether a memory contains user preferences, facts, or events
- whether a memory appears to be a recommendation or answer

These features must be general-purpose and not derived from MemGate test-set artifacts.

## 4. Disallowed Methods

The following methods are not allowed for compliant submissions.

### 4.1 MemGate-specific prompt-template rules

Do not write rules that recognize MemGate prompt families or surface forms.

Disallowed examples:

```python
if "hidden value" in user_input:
    use_rank_plan_A()

if "unlabeled value" in user_input:
    use_rank_plan_B()

if "private field" in user_input:
    inject()

if "mockup" in user_input:
    do_not_inject()
```

Rules of this kind are benchmark-specific prompt recognition, not general memory reasoning.

### 4.2 Hand-coded field-to-domain expansion tables

Do not manually encode MemGate-specific mappings from visible fields to target-memory domains.

Disallowed example:

```python
EXPANSIONS = {
    "education": "degree graduated university major",
    "device": "laptop ram memory upgrade",
    "pet profile": "cat dog pet breed vet",
    "morning logistics": "commute work travel time",
}
```

A general query rewriting model is allowed. A manually curated expansion dictionary derived from the benchmark is not.

### 4.3 Fixed benchmark name lists

Do not write rules based on fixed names or roles observed in the benchmark.

Disallowed examples:

```python
if "Pat's" in user_input or "Jordan's" in user_input:
    do_not_inject()

THIRD_PARTY_NAMES = ["Pat", "Jordan", "Lee", "Casey", "Avery"]
```

General named-entity recognition and ownership reasoning are allowed. Hardcoded benchmark name lists are not.

### 4.4 Exact-match, lookup-table, or memorization solvers

Do not write rules that match known prompts, prompt hashes, n-gram signatures, or sample-specific strings.

Disallowed examples:

```python
if user_input == "The shared profile sheet has my row blank; fill the value.":
    return True

if hash(user_input) in KNOWN_REQUIRED_HASHES:
    return True
```

### 4.5 Test-set tuning

Do not manually tune a system after inspecting hidden test-set failures.

Allowed:

- development on a public dev set
- ablations on a public dev set
- general algorithmic improvements before final test evaluation

Disallowed:

- repeatedly running on the hidden test set and adding rules for observed failures
- manually adding prompt-family rules after reviewing hidden test examples
- tuning expansion dictionaries, name lists, or gating patterns against hidden test prompts

### 4.6 Metadata leakage

Do not use benchmark metadata in any way.

Disallowed examples:

```python
if sample["probe_type"] == "required":
    inject()

if memory.get("is_target_memory"):
    select(memory)

query = sample["source_question"]
```

## 5. Compliant vs. Artifact-Exploit Systems

MemGate distinguishes between two categories of systems.

### 5.1 Compliant systems

A compliant system follows all rules above and is intended to measure generalizable memory capability.

Only compliant systems should be used for main benchmark comparisons.

### 5.2 Artifact-exploit baselines

Artifact-exploit baselines are systems that intentionally exploit benchmark artifacts, such as prompt templates, fixed field vocabularies, fixed name lists, or repeated test-set patterns.

They are useful for auditing dataset weaknesses, but they are not valid evidence of general proactive memory capability.

A handcrafted benchmark-specific solver should be reported separately as an artifact-exploit baseline, not as a compliant submission.

## 6. Submission Checklist

A compliant submission should include a signed or explicit statement confirming:

```text
[ ] The system uses only user_input and memory text at runtime.
[ ] memory_id is used only as an opaque handle for returning selected memories.
[ ] The system ignores probe_type, gold labels, diagnostics, source_question, and all other benchmark metadata.
[ ] The system does not contain MemGate-specific prompt-template rules.
[ ] The system does not contain a hand-coded field-to-domain expansion table derived from MemGate.
[ ] The system does not contain fixed benchmark name lists.
[ ] The system does not use exact-match, hash-match, or lookup-table prompt memorization.
[ ] The system was not tuned on hidden test-set failures.
[ ] The same system can run unchanged on non-MemGate user-memory data.
```

## 7. Recommended Reporting

Reports should include at least:

- Required target recall@k
- Gate accuracy
- Gate precision
- Gate recall
- False injection rate
- Target selected rate
- Forbidden selected rate

Do not report only a single aggregate score. MemGate intentionally separates retrieval, gating, and forbidden-memory safety.

Recommended baseline categories:

```text
Compliant baselines:
- BM25 + generic gate
- Hybrid retrieval + generic gate
- LLM query constructor + generic gate
- LLM gate over retrieved memories

Artifact-exploit baselines:
- template-rule solvers
- fixed field-expansion solvers
- fixed name-list solvers
- exact-match solvers
```

## 8. Interpretation Notes

A high gate score alone does not prove proactive memory capability if the system relies on benchmark-specific surface patterns.

A strong MemGate system should satisfy all of the following:

1. Improve required target recall under query gap.
2. Inject memory when the current request genuinely depends on the user's own prior context.
3. Avoid injecting retrieved memories for neutral or scope-mismatched requests.
4. Avoid selecting forbidden memories even when they are retrieved.
5. Use generalizable reasoning that applies outside the MemGate dataset.

## 9. Short Version

```text
Allowed:
  General retrieval, query rewriting, reranking, LLM gates, learned gates, and general privacy/scope rules.

Disallowed:
  MemGate-specific templates, hand-coded field expansion dictionaries, fixed benchmark name lists, exact prompt matching, metadata leakage, and tuning on hidden test failures.

Main principle:
  Build a memory system, not a MemGate solver.
```
