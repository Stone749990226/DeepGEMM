# 11 · Heuristic 重标定与 FP8 数值精度

> **难点强度：★★★☆☆**（表格工程 + 细致测试）

## 1. Heuristic 的重新标定

SM100 Mega MoE 的 heuristic（[csrc/jit_kernels/heuristics/mega_moe.hpp](csrc/jit_kernels/heuristics/mega_moe.hpp) + [analysis §5.2](reports/mega_moe_code_analysis.md)）决定三件事：

1. **Block 配置**（BLOCK_M、STORE_BLOCK_M、epilogue warpgroups）
2. **Expert Wave**（num_experts_per_wave）
3. **Pipeline 深度**（num_stages）

每一项在 Hopper 上都要重新取值。

### 1.1 Block 配置候选集

**SM100 候选**（BLOCK_M）：`{16, 32, 64, 96, 128, 192}`

**Hopper 候选建议**：`{16, 32, 64, 96, 128}` —— 移除 192。

原因（来自 02/08 章）：
- 192 的累加器在 WGMMA 寄存器里装不下（每 thread 需要 384 FP32，超过 256 reg/thread 上限）。
- 192 对 BLOCK_M 是 WGMMA `m64n_k32` 的 3 次迭代，warpgroup 内 stacking 3 次；128 只需 2 次，结构更简单。

WGMMA 选择：
- `BLOCK_M = 16/32` → 只能用 `wgmma.m64n_k32`，M 维 padding 浪费多（但 MoE 路由本就稀疏，padding 难免）。
- `BLOCK_M = 64` → `wgmma.m64n_k32` 1 次。
- `BLOCK_M = 96` → 需要 `m64n_k32 × 1 + m32n_k16`? 实际上 WGMMA 只提供 m64，没有 m32 单独指令。96 在 Hopper 上用 `m64n_k32` × 2 做 128 再丢弃尾部，等效 BLOCK_M=128，候选里可以剔除。
- `BLOCK_M = 128` → `m64n_k32 × 2`。

**修订后 Hopper 候选**：`{16, 32, 64, 128}`

对应 LCM = 64（原 SM100 LCM=384），workspace pool 容量的对齐常数相应改小。

### 1.2 `BLOCK_M` 选择阈值重标

SM100 的阈值（expected tokens/expert）：
```
≤ 8.5  → BLOCK_M=16
≤ 16.5 → BLOCK_M=32
≤ 32.5 → BLOCK_M=64
≤ 64.5 → BLOCK_M=96       # Hopper 去掉
≤ 96.5 → BLOCK_M=128
> 96.5 → BLOCK_M=192       # Hopper 改成 128
```

Hopper 建议：
```
≤ 8.5  → BLOCK_M=16
≤ 16.5 → BLOCK_M=32
≤ 32.5 → BLOCK_M=64
> 32.5 → BLOCK_M=128
```

阈值具体边界要 **跑 benchmark 扫表** 确认（见 §4 测试矩阵）。

### 1.3 Expert Wave 数

`num_experts_per_wave` 的公式保持不变，但其中 `num_l1_blocks_per_expert = num_m_blocks × num_n_blocks` 的 `num_n_blocks` 变化：
- SM100: `2 × intermediate_hidden / BLOCK_N`（N 有 gate+up 两半）
- Hopper: 同样 `2 × intermediate_hidden / BLOCK_N`（权重仍是 gate/up 合并的）

结构不变。但 Hopper 的 `num_sms` 是 **132 (H100) 或 144 (H200)**，SM100 的是 **148 (GB200)**，这会让 `2 × num_sms / num_l1_blocks_per_expert` 的结果略小，`num_experts_per_wave` 倾向于略小。

### 1.4 Pipeline 深度

见 09 章。Hopper 单 CTA 配置的 stage 数在 ~4-8，cluster=2 + multicast 下恢复到 ~10。代码里要求 `num_stages ≥ 2`，Hopper 容易满足；但 `num_stages ≥ 4` 是性能下限，heuristic 里可以设 `max(num_stages_from_smem, 4)` 保底，如果 smem 不够就退一档 BLOCK_M。

## 2. FP8 数值精度

### 2.1 算子链累积误差

Mega MoE forward 一次要经历：
```
FP8 x   ×   FP8 w1   → FP32 accum (L1)
FP32 accum × SF_l1_acts × SF_l1_w  (scale promotion)
SwiGLU(FP32)
× topk_weight(FP32)
FP32 → FP8 量化（amax 归一）
FP8 y   ×   FP8 w2   → FP32 accum (L2)
FP32 accum × SF_l2_acts × SF_l2_w (scale promotion)
FP32 → BF16 cast
跨 rank BF16 写回
FP32 累加 top-k（final combine）
FP32 → BF16 输出
```

**关键精度风险点**：
1. **L1 SwiGLU 后的 FP8 量化**：amax scale 以 per-row（或 per-32-channel）做，精度依赖 clamp 和饱和。SM100 Mega MoE 支持 `kActivationClamp`，Hopper 版本要保持相同语义。
2. **SF 精度**：UE8M0（2^e）会引入 "scale 对齐" 损失。从 FP4 用 UE8M0 `rk=32` → FP8 用 float `rk=128` 的变化，scale 粒度变粗，单个 scale 覆盖的 K 范围变大，误差**可能增大**。具体影响要 benchmark 验证。
3. **CUDA promotion vs 硬件 block-scaled MMA**：浮点运算顺序不同（先乘 scale 再累加 vs 硬件 fused），bitwise 不等几乎必然。

### 2.2 测试策略：tolerance 替代 bitwise

原 `tests/test_mega_moe.py` 要求 fused 与 baseline `bitwise_equal`。Hopper port 需要放宽到：
```python
assert_close(fused_y, baseline_y, atol=1e-2, rtol=5e-3)
```
`atol=1e-2` 在 BF16 输出规模下是 ~0.5% 的绝对精度，对 MoE 合理。

但**要保留 bitwise 一致性测试**在另一个维度：**同一份 Hopper fused kernel 运行两次**应 bitwise equal（确定性）。

### 2.3 测试矩阵

| 维度 | 候选 |
|---|---|
| Batch size (tokens/rank) | 8, 32, 128, 512 |
| Top-k | 4, 6, 8 |
| num_experts | 128, 256 |
| Hidden | 5120, 7168 |
| Intermediate hidden | 2048, 4096 |
| EP size | 4, 8, 16 |
| 不均衡 mask ratio | 0%, 20%, 50% |

每个配置对比：
- **bitwise self-consistency**：同一 kernel 跑两次 output 一致
- **numerical closeness**：与 BF16 baseline（即 `tests/test_mega_moe.py` legacy path，权重 dequant 后 BF16 GEMM）`assert_close(atol=1e-2)`
- **performance**：TFLOPS, HBM GB/s, NVLink GB/s, 相对 legacy 加速比

### 2.4 Numerical gotchas 清单

- **fast_math 路径**：SM100 的 SwiGLU 有 `__expf` / `__frcp_rn` 快速近似路径 (kActivationClamp 非 inf 时)，精度有 ~1e-4 误差。Hopper 版本建议同样支持，保持 API 一致。
- **clamp**：SM100 里 clamp 是**先 bf16 cast 再 clamp**，精度只到 bf16。Hopper 保持同样做法，结果相同。
- **topk_weight 应用位置**：必须在 L1 epilogue 量化**之前**应用。换顺序会导致每 top-k 分支的量化 scale 不同，combine 后结果错。
- **FP8 saturation**：SwiGLU 的值可能超过 FP8 E4M3 的 max（448）。amax-based scale 保证不溢出，但如果 amax 计算本身出错会 silently corrupt。调试时加个 FP32 中间输出比对。

## 3. 性能预期

### 3.1 吞吐估算

H100 SXM5:
- FP8 Tensor Core: 2 PFLOPS (with sparsity 4 PFLOPS)
- HBM3 BW: 3 TB/s
- NVLink 4.0: 900 GB/s

**计算 bound 区域**：batch 大（L1 M 维 > 64），FP8 GEMM 可达 60-70% 峰值 ≈ 1.2-1.4 PFLOPS。
**带宽 bound 区域**：batch 小（token 少于 expert 数量 × rank 数），权重读取是瓶颈，~2.5 TB/s 实测可用。
**通信 bound 区域**：极小 batch（< 32 token/rank），dispatch/combine NVLink 占主导，~700 GB/s 可用。

SM100 Mega MoE 的典型 TFLOPS 是 ~1.5-2.0 (FP4 × FP8)，Hopper FP8 × FP8 等效 TFLOPS 预计 ~0.6-1.0，**大约为 SM100 的 40-50%**。

### 3.2 相对 legacy 的加速比

相对 baseline（分阶段 DeepEP + DeepGEMM + TileLang + DeepEP）：
- SM100 Mega MoE 在 H-wave 实测 ~1.5-3× 加速。
- Hopper port 预期 ~1.2-2× 加速（overlap 能力略差，但 launch 开销节省等收益不变）。

这足以证明移植价值。

## 4. 测试基础设施改造

`tests/test_mega_moe.py` 的 baseline 生成路径要改：
```python
if is_sm90():
    # 不再调用 m_grouped_fp8_fp4_gemm_nt_contiguous
    # 改为 m_grouped_fp8_fp8_gemm_nt_contiguous（DeepGEMM 已有 sm90 实现）
    ...
```
这要求仓库 exposes `m_grouped_fp8_gemm` 的 Python binding（已存在），并且新的 Mega MoE SM90 kernel 的权重 transform 与之对齐（方案 A，见 04 章）。

## 5. 小结

- Heuristic 的 BLOCK_M 候选集从 `{16,32,64,96,128,192}` 改为 `{16,32,64,128}`，LCM 相应改小。
- 阈值、wave 公式结构不变，数值需要扫表重新标定。
- **数值上 fused 与 legacy 不可能 bitwise equal**，改用 `assert_close(atol=1e-2)`。
- FP8 SwiGLU 量化、SF 粒度变化是主要精度风险，需要新的测试矩阵。
- 预期 Hopper 版本吞吐 ~SM100 的 40-50%，相对 legacy 加速 ~1.2-2×。
