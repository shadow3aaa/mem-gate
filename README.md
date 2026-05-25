# MEM-GATE

MEM-GATE 是一个基准测试，用于测试agent记忆模块的能力，专门针对记忆自觉和非直接检索场景。

## 运行

这会运行默认的bm25基准测试。如需要评测自己的实现，请参考[main.py](main.py)中的 `MySystem` 类。

```bash
uv sync
uv run main.py
```

## 评测结果

运行 `uv run main.py` 后，评测结果将以表格形式显示在控制台中，并保存到指定文件。下面是一个示例输出：

```text
MemGate Eval: 100%|█| 250/250 [00:06<00:00, 38.97sample/s, e8a79c70_availability_without [availability_without_memory]]

MemGate Summary
====================================================
dataset=datas/memgate-eval.jsonl | samples=250 | top_k=10 | stage3=off

Metric                Value
------------------ --------
Gate Acc              40.0%
Overall R@10          18.4%
Overall R@5           13.6%
Overall R@1            4.0%
Gate=True R@10        46.0%
Gate=True R@5         34.0%
Gate=True R@1         10.0%

records=results\runs\my_system.jsonl
```

## RULES

评测实现必须满足[RULES.md](RULES.md)。否则应该认为无效。
