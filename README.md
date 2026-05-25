# MEM-GATE

MEM-GATE 是一个基准测试，用于测试agent记忆模块的能力，专门针对记忆自觉和非直接检索场景。

## 运行

这会运行默认的bm25基准测试。如需要评测自己的实现，请参考[main.py](main.py)中的 `MySystem` 类。

```bash
uv sync
uv run main.py
```
