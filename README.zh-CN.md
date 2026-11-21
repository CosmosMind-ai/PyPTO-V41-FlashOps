# PyPTO-V41-FlashOps

[English](README.md) | **中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> CosmosMind 昇腾 PyPTO 算子系列之一 —— 全部算子由 [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1) 驱动。

面向 **昇腾 910C** 的 DeepSeek V4.1-Flash 风格算子库，使用
[PyPTO](https://github.com/hw-native-sys/pypto) 编写：四个融合算子覆盖现代线性注意力 /
稀疏注意力推理链路 —— 多头上下文混合（Mega-mHC）、条件记忆（Engram）、精确层级稀疏索引
（HierIndex）、滑窗有界重放 KV 投影 + RoPE（SWA-Replay）。

**为什么建这个库。** 我们在 910C（`a2a3`）上实测了 PyPTO 官方 V4.1 模型路径，发现关键组件
在该芯片上要么缺失、要么无法运行：官方 HC sinkhorn 路径基于 a5 ISA 的 extern C++，在 910C
上无法编译；官方 Engram 在原生模型宽度下超出 910C 的向量缓冲上限；官方 `decode_swa` 依赖
比 910C 软件栈更新的 PyPTO API。本库用今天就能在 910C 上编译运行的算子补上这些缺口，
每个算子都由独立 golden 参考与密封留出集（sealed holdout）双重把关。完整证据见
[`docs/OFFICIAL-ALIGNMENT.md`](docs/OFFICIAL-ALIGNMENT.md)。

## 算子

| 算子 | 功能 | 密封留出集延迟 | 官方 PyPTO @ 910C |
|---|---|---|---|
| [`mega_mhc`](kernels/mega_mhc/) | 融合 mHC 层步进：Sinkhorn 组合矩阵 + 门控残差更新 + 下一层系数预测（M=4, D=256） | **61.1 µs** @ T=2048/4096 bf16 | 不可运行 —— 官方 `hc.py` 使用 a5 专用 extern C++ |
| [`engram`](kernels/engram/) | Engram 条件记忆：哈希 gather → 双路投影 → 双 RMS 归一化 → 带符号平方根门控 | **57.6 / 119.6 / 337.3 µs** @ T=64/256/1024 | 204.2 / 497.8 / 1428.8 µs → **快 3.5× / 4.2× / 4.2×** |
| [`hierindex`](kernels/hierindex/) | 精确层级稀疏索引：FP32 打分 → 块最大值 → top-256 块 → 池内精确有序 top-512 | **83.8 µs** @ S=131072 | 互补算子类 —— 官方 `decode_indexer` 面向压缩缓存 |
| [`swa_replay`](kernels/swa_replay/) | SWA 有界重放 KV 投影（D=7168→1024）+ 绝对位置 RoPE | **465.6 µs** @ B=32（4096 行） | 不可运行 —— 官方 `decode_swa` 依赖更新的 PyPTO API |

所有延迟均为真机昇腾 910C（`a2a3`）上的逐轮有效耗时中位数，测量协议见
[`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md)。每个算子都通过了**密封留出集**
（全新形状，正确性 + 延迟双门禁），对照独立的 PyTorch golden 参考。

## 目录结构

```
kernels/
  mega_mhc/kernel.py    # 融合 mHC 层步进
  engram/kernel.py      # Engram 条件记忆前向核心
  hierindex/kernel.py   # 精确层级稀疏索引
  swa_replay/kernel.py  # SWA 有界重放 KV 投影 + RoPE
benchmarks/BENCHMARKS.md
docs/OFFICIAL-ALIGNMENT.md
```

## 使用方式

每个算子是一个自包含的 `kernel.py`，遵循 pypto-lib `examples/intermediate/` 风格，
暴露标准 build 契约：

```python
fn, specs, golden_fn, rtol, atol = build(shape)
```

- `fn` —— `@pl.jit` 算子或已构建的 `@pl.program`，经 PyPTO 编译；
- `specs` —— 与算子签名对应的 `golden.TensorSpec` / `ScalarSpec`，含输入分布；
- `golden_fn` —— 就地填充输出的独立 PyTorch 参考实现；
- `rtol`、`atol` —— 校验容差。

依赖：昇腾 910C（`a2a3`）、CANN ≥ 9.0、`pypto`、`pypto-lib`（其 `golden` 包提供
经过验证的启动路径）、`torch`。示例：

```python
import importlib.util

spec = importlib.util.spec_from_file_location("engram", "kernels/engram/kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

fn, specs, golden_fn, rtol, atol = mod.build({"T": 256, "dtype": "bf16"})
# 通过 pypto-lib 的 golden runner 编译并下发 fn，再按 (rtol, atol) 对照 golden_fn 校验
```

每个算子的精确语义、张量接口与支持域见各自目录下的 README。不支持的形状会直接抛错，
算子不会在契约外静默计算。

## 状态

v0.1.0 —— 四个算子均在 910C 上通过正确性门禁与密封留出集。接口遵循 V4.1 算子语义；
凡官方 PyPTO  counterpart 存在且能在 910C 运行的，我们都给出同卡同协议的对照数据
（[`docs/OFFICIAL-ALIGNMENT.md`](docs/OFFICIAL-ALIGNMENT.md)）。本库是算子库而非完整
模型路径：paged-KV scatter、哈希计算与主机侧编排刻意不在范围内。

## 许可证

[MIT](LICENSE)
