# Mega MoE 代码实现逻辑分析

本文基于当前仓库源码梳理 DeepGEMM 中 Mega MoE 的实现逻辑。相关入口包括：

- Python API: `deep_gemm/mega/__init__.py`
- C++/pybind API: `csrc/apis/mega.hpp`
- JIT wrapper: `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`
- heuristic: `csrc/jit_kernels/heuristics/mega_moe.hpp`
- workspace/layout: `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`
- device scheduler: `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
- 核心 CUDA kernel: `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
- 测试和 benchmark: `tests/test_mega_moe.py`

## 1. Mega MoE 的目标

README 中对 Mega MoE 的定位是：把 EP dispatch、linear 1、SwiGLU、linear 2、EP combine 融合到一个 mega-kernel 中，并让 NVLink 通信和 Tensor Core 计算重叠。

传统 baseline 在 `tests/test_mega_moe.py` 里是分阶段执行的：

1. `deep_ep.ElasticBuffer.dispatch` 做 EP dispatch，把本 rank 的 token 按 `topk_idx` 发送/展开到对应专家所在 rank。
2. `m_grouped_fp8_fp4_gemm_nt_contiguous` 做第一层 grouped GEMM。
3. `tilelang_ops.swiglu_apply_weight_to_fp8` 做 SwiGLU、乘 top-k weight、量化为 FP8。
4. 再调用一次 grouped GEMM 做第二层。
5. `ep_buffer.combine` 把 top-k 专家输出 combine 回原 token。

Mega MoE 将上述流程融合到一个 kernel：dispatch warp 负责跨 rank 拉取 token，MMA 相关 warp 做 L1/L2 FP8xFP4 GEMM，epilogue warp 同时完成 SwiGLU、FP8 量化、L2 输出写回远端 combine buffer 和最终 reduce combine。

### 端到端数据流总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Mega MoE Fused Kernel 数据流                         │
│                                                                             │
│  Rank 0                          Rank 1                          Rank N     │
│  ┌──────────┐                    ┌──────────┐                    ┌────────┐ │
│  │ x, topk  │                    │ x, topk  │                    │x, topk │ │
│  │ (symm    │                    │ (symm    │                    │(symm   │ │
│  │  buffer) │                    │  buffer) │                    │ buffer)│ │
│  └────┬─────┘                    └────┬─────┘                    └───┬────┘ │
│       │                               │                              │      │
│       ▼                               ▼                              ▼      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 1: Dispatch (dispatch warps)                                     │ │
│  │  1. 统计本 rank 各 expert 的 token 数 (atomicAdd_block)                │ │
│  │  2. 全局 atomic 汇总到 expert_send_count                              │ │
│  │  3. 写 src_token_topk_idx 到目标 rank 的 workspace (NVLink)           │ │
│  │  4. Grid sync + SM0 汇总 expert_recv_count/sum                        │ │
│  │  5. NVLink barrier: 所有 rank 完成写入                                │ │
│  │  6. Pull: 从源 rank 拉取 FP8 token + SF + topk_weight 到本地 L1 pool  │ │
│  │     (round-robin min-peeling 均衡跨 rank 访问)                        │ │
│  │  7. 写 TokenSrcMetadata, 更新 l1_arrival_count                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 2: L1 GEMM (MMA warps, per expert wave)                         │ │
│  │  - TMA load L1 acts (FP8) + weights (FP4) + scale factors             │ │
│  │  - UMMA: FP8 x FP4 矩阵乘, accumulator 在 tensor memory              │ │
│  │  - 等待条件: l1_arrival_count[block] == valid_m                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 3: L1 Epilogue (epilogue warps)                                  │ │
│  │  - 从 tensor memory 读 accumulator                                     │ │
│  │  - SwiGLU: silu(gate) * up * topk_weight                              │ │
│  │  - Amax reduction → FP8 E4M3 量化                                     │ │
│  │  - TMA store 到 l2_acts (= L2 输入)                                   │ │
│  │  - 写 SF 到 l2_sf_buffer (UTCCP 布局)                                 │ │
│  │  - 设置 l2_arrival_mask bit                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 4: L2 GEMM (MMA warps, same expert wave)                        │ │
│  │  - 等待条件: l2_arrival_mask[block] == expected (所有 L1 N blocks)     │ │
│  │  - TMA load L2 acts (FP8) + weights (FP4) + scale factors             │ │
│  │  - UMMA: FP8 x FP4 矩阵乘                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 5: L2 Epilogue (epilogue warps)                                  │ │
│  │  - 从 tensor memory 读 accumulator → cast BF16                        │ │
│  │  - 写到 shared memory (swizzled 布局)                                  │ │
│  │  - 读 TokenSrcMetadata → 得到 (dst_rank, dst_token, dst_topk)         │ │
│  │  - 通过 sym_buffer.map 写到目标 rank 的 combine_token_buffer (NVLink) │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 6: NVLink Barrier + Combine Reduce (epilogue warps)              │ │
│  │  - NVLink barrier: 确保所有 rank 的 L2 写回完成                       │ │
│  │  - 每个 rank 本地遍历自己的 token                                      │ │
│  │  - 对每个有效 topk slot, TMA load combine_buffer[slot, token] (BF16)  │ │
│  │  - FP32 累加所有 topk 分支 → cast BF16 → TMA store 到 y[token]       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 7: Workspace 清理 (dispatch warps, 与 combine 重叠)              │ │
│  │  - SM0 清 expert_send_count                                            │ │
│  │  - 其他 SM 清 recv_count/sum, l1_arrival, l2_arrival                   │ │
│  │  - 累加 cumulative_local_expert_recv_stats                             │ │
│  │  - NVLink barrier: 确保所有 rank 清理完成                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Host 侧调用链

### 2.1 Python 入口

`deep_gemm/mega/__init__.py` 暴露三个主要函数：

- `get_symm_buffer_for_mega_moe(...)` — 分配 symmetric buffer 并切分视图
- `transform_weights_for_mega_moe(...)` — 权重布局转换
- `fp8_fp4_mega_moe(...)` — 启动 fused kernel

使用方式（伪代码）：

```python
# 1. 分配 symmetric buffer（自动对齐 num_max_tokens_per_rank 到 384）
buffer = get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens, num_topk, hidden, inter_hidden)

# 2. 将输入 copy 到 buffer 的视图
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# 3. 权重布局转换（只需做一次）
l1_weights, l2_weights = transform_weights_for_mega_moe(l1_weights, l2_weights)

# 4. 启动 fused kernel
fp8_fp4_mega_moe(y, l1_weights, l2_weights, buffer)
```

`get_symm_buffer_for_mega_moe` 会把 `num_max_tokens_per_rank` 对齐到 `_C.get_token_alignment_for_mega_moe()`。C++ 里这个 alignment 是 `layout::kLCMCandidateBlockM = 384`，原因是 Mega MoE 会根据不同 token 数选择 `BLOCK_M`，候选值包括 `{8, 16, 32, 64, 96, 128, 192}`，384 是这些候选值的最小公倍数，便于 workspace pool 的统一容量规划。

### 2.2 symmetric memory

`SymmBuffer` 使用 `torch.distributed._symmetric_memory.empty` 分配一段每个 rank 地址可互相映射的 buffer，再通过 `symm_mem.rendezvous` 得到跨 rank buffer 指针列表。分配后立即 `zero_()` 并 barrier 同步，确保所有 rank 的 workspace 初始状态一致。

Python 层将 raw int8 buffer 切成 8 个视图（C++ 侧 `get_symm_buffer_size_for_mega_moe` 计算偏移）：

```
Symmetric Buffer 内存布局（单个 rank 视角）:
┌──────────────────────────────────────────────────────────────────────┐
│ Workspace (元数据区)                                                 │
│  ├─ barrier signals (32 bytes)                                       │
│  ├─ expert_send_count[num_experts] (uint64)                          │
│  ├─ expert_recv_count[num_experts] (uint64)                          │
│  ├─ expert_recv_count_sum[num_experts_per_rank] (uint64)             │
│  ├─ l1_arrival_count[num_max_pool_blocks] (uint32, padded to even)   │
│  ├─ l2_arrival_mask[num_max_pool_blocks] (uint64)                    │
│  ├─ src_token_topk_idx[experts_per_rank][ranks][max_recv] (uint32)   │
│  └─ TokenSrcMetadata[num_max_pool_tokens] (3 x uint32)              │
├──────────────────────────────────────────────────────────────────────┤
│ x:            FP8 E4M3  [num_max_tokens_per_rank, hidden]            │
│ x_sf:         int       [num_max_tokens_per_rank, hidden/128]  K-maj │
│ topk_idx:     int64     [num_max_tokens_per_rank, num_topk]          │
│ topk_weights: float     [num_max_tokens_per_rank, num_topk]          │
├──────────────────────────────────────────────────────────────────────┤
│ l1_acts:      FP8 E4M3  [num_max_pool_tokens, hidden]               │
│ l1_acts_sf:   int       [num_padded_sf_pool_tokens, hidden/128] M-maj│
│ l1_topk_wt:   float     [num_max_pool_tokens, 1]                    │
├──────────────────────────────────────────────────────────────────────┤
│ l2_acts:      FP8 E4M3  [num_max_pool_tokens, intermediate_hidden]   │
│ l2_acts_sf:   int       [num_padded_sf_pool_tokens, inter_hidden/128]│
├──────────────────────────────────────────────────────────────────────┤
│ combine_buf:  BF16      [num_topk, num_max_tokens_per_rank, hidden]  │
└──────────────────────────────────────────────────────────────────────┘
```

注意 `x_sf` 是 K-major（contiguous in K），而 `l1_acts_sf` 和 `l2_acts_sf` 是 M-major（stride 为 `{1, num_padded_sf_pool_tokens}`），这是因为 UTCCP 从 shared memory copy SF 到 tensor memory 时需要 MN-major 布局。

### 2.3 C++ API 校验

`csrc/apis/mega.hpp` 的 `fp8_fp4_mega_moe` 做了关键约束检查：

- 只支持 recipe `(1, 1, 32)`，即 `rm=1, rn=1, rk=32`（SM100 MMA recipe 参数）。
- 当前只支持 activation `"swiglu"`。
- l1/l2 权重必须是 K-major FP4 grouped 权重，且 contiguous。
- `l1_weights` shape 对应 `[num_experts_per_rank, 2 * intermediate_hidden, hidden]`。
- `l2_weights` shape 对应 `[num_experts_per_rank, hidden, intermediate_hidden]`。
- `hidden` 和 `intermediate_hidden` 必须是 128 的倍数（SF 布局要求）。
- 权重 SF 使用 UE8M0 packing 格式，MN-major 布局，需满足 TMA 对齐。
- symmetric buffer 大小必须至少等于 `get_symm_buffer_size_for_mega_moe` 计算出的大小。
- 仅 `arch_major == 10` 会 dispatch 到 `sm100_fp8_fp4_mega_moe`，也就是当前实现面向 Blackwell SM100。
- 可选的 `cumulative_local_expert_recv_stats` 必须是 int32、大小等于 `num_experts_per_rank`、contiguous。

如果设置了 `DG_COMM_KERNEL_DEBUG`，每次 kernel 后会把整块 symmetric buffer 清零。测试里因此每次调用前都重新 copy 输入。

## 3. 权重转换逻辑

`transform_weights_for_mega_moe` 做两类转换。

### 3.1 L1 权重 interleave gate/up

原始 L1 权重 N 维度是 `[gate | up]`，即前半是 gate，后半是 up。Mega MoE 的 L1 epilogue 使用 `SM100_TMEM_LOAD_16dp256b1x` 从 tensor memory 加载 accumulator，该指令的数据排列使得 gate/up 对为 `(values[0], values[2]), (values[1], values[3]), (values[4], values[6]), (values[5], values[7])`。为了让这种硬件排列直接对应 gate/up 配对，Python 侧先按 granularity 8 做 interleave：

```python
# _interleave_l1_weights 的逻辑（gran=8）:
# 原始: [gate_0..gate_{N/2-1} | up_0..up_{N/2-1}]
# 转换: [gate_0..7, up_0..7, gate_8..15, up_8..15, ...]

def interleave(t, gran=8):
    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up   = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(g, n, *rest)
```

对 L1 权重和 L1 权重 SF 都做同样的 interleave。

### 3.2 Scale Factor UTCCP 转置

L1 和 L2 的 scale factor 都调用 `_transpose_sf_for_utccp`。该函数把 `[num_groups, mn, packed_sf_k]` 在每 128 个 MN 元素内做 4×32 转置：

```python
def _transpose_sf_for_utccp(sf):
    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    # 每 128 个元素视为 [4, 32]，转置为 [32, 4]
    return (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
              .transpose(2, 3)
              .reshape(num_groups, mn, packed_sf_k))
```

这个转置匹配 SM100 UTCCP（`SM100_UTCCP_4x32dp128bit_2cta`）把 scale factor 从 shared memory copy 到 tensor memory 的访问模式。kernel 侧有对应的 `transform_sf_token_idx` lambda 做相同的索引变换：

```cpp
// kernel 中 dispatch warp 写 SF 时的索引变换
transform_sf_token_idx = [](uint32_t token_idx_in_expert) {
    uint32_t idx = token_idx_in_expert % BLOCK_M;
    return token_idx_in_expert / BLOCK_M * SF_BLOCK_M
         + (idx & ~127u)           // 128 对齐的基地址
         + (idx & 31u) * 4         // 32-group 内的行 × 4
         + ((idx >> 5) & 3u);      // 4-group 内的列
};
```

## 4. Workspace 和 Pool 布局详解

`deep_gemm/include/deep_gemm/layout/mega_moe.cuh` 定义了 `Workspace`、`Data`、`Buffer` 三个核心结构体。

### 4.1 Workspace 内存布局

Workspace 位于 symmetric buffer 的最前端，包含 kernel 运行所需的所有元数据：

```
Workspace 内存布局（字节偏移）:
┌─────────────────────────────────────────────────────────────────┐
│ [0..31] Barrier Signals (32 bytes, 固定)                        │
│   ├─ [0..15]  4 × uint32 grid sync counters                    │
│   ├─ [16..19] uint32 NVLink barrier counter                     │
│   └─ [20..27] 2 × int NVLink barrier signals (phase 0/1)       │
├─────────────────────────────────────────────────────────────────┤
│ expert_send_count[num_experts]           (uint64 × num_experts) │
│   高32位: 完成标记计数, 低32位: token数                          │
├─────────────────────────────────────────────────────────────────┤
│ expert_recv_count[num_ranks × experts_per_rank]  (uint64)       │
│   索引: [rank_idx * experts_per_rank + expert_idx]              │
│   存储: 源 rank 发给本 rank 某 expert 的 token 数               │
├─────────────────────────────────────────────────────────────────┤
│ expert_recv_count_sum[num_experts_per_rank]       (uint64)      │
│   高32位: 完成状态 (== kNumSMs * kNumRanks 时表示就绪)          │
│   低32位: 该 expert 总接收 token 数                              │
├─────────────────────────────────────────────────────────────────┤
│ l1_arrival_count[num_max_pool_blocks]    (uint32, padded even)  │
│   dispatch 写入一个 token 后 +1, MMA 等到 == valid_m 才开始     │
├─────────────────────────────────────────────────────────────────┤
│ l2_arrival_mask[num_max_pool_blocks]     (uint64)               │
│   L1 epilogue 完成一个 N block 后设置对应 bit                   │
│   L2 TMA load 等到所有需要的 bit 都 ready                       │
├─────────────────────────────────────────────────────────────────┤
│ src_token_topk_idx[experts_per_rank][num_ranks][max_recv]       │
│   (uint32) dispatch 阶段写入, 记录源 token_topk_idx             │
│   用于目标 rank 反查源 token 位置                                │
├─────────────────────────────────────────────────────────────────┤
│ TokenSrcMetadata[num_max_pool_tokens]    (3 × uint32 each)      │
│   { rank_idx, token_idx, topk_idx }                             │
│   L2 epilogue 用于写回 combine buffer 时定位目标                 │
├─────────────────────────────────────────────────────────────────┤
│ (对齐到 16 bytes)                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Pool 容量计算

`get_num_max_pool_tokens` 估算 local expert token pool 上限：

```cpp
// 伪代码
num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
num_max_experts_per_token = min(num_topk, num_experts_per_rank);
pool_capacity = align(
    num_max_recv_tokens * num_max_experts_per_token
    + num_experts_per_rank * (kMaxCandidateBlockM - 1),  // 每个 expert 最多 191 的 padding
    kLCMCandidateBlockM  // 对齐到 384
);
```

这里 `num_max_experts_per_token` 取 `min(num_topk, num_experts_per_rank)` 是因为一个 token 最多路由到 `num_topk` 个 expert，但本 rank 只有 `num_experts_per_rank` 个 expert，所以实际最多命中两者的较小值。

`num_max_pool_blocks = num_max_pool_tokens / kMinCandidateBlockM`（即除以 8），这是 `l1_arrival_count` 和 `l2_arrival_mask` 数组的大小上界。

### 4.3 SF Pool 容量

```cpp
num_padded_sf_pool_tokens = (num_max_pool_tokens / block_m) * align(block_m, 128);
```

由于 UTCCP 要求 128 元素对齐，每个 BLOCK_M 大小的 block 在 SF 维度上被 pad 到 128 的倍数。C++ 侧会遍历所有 7 个候选 BLOCK_M 取最大值，确保任何配置都能容纳。

### 4.4 Combine Buffer

L2 输出不是直接写 `y`，而是先写入 symmetric buffer 尾部的 `combine_token_buffer`：

```
combine_token_buffer: BF16 [num_topk, num_max_tokens_per_rank, hidden]
```

每个 expert 分支的 L2 输出会根据 `TokenSrcMetadata` 写回原始 token 所在 rank 的对应 top-k slot。等所有 rank 都写完后，再由原 token 所在 rank 本地 reduce top-k slots，得到最终 `y[token]`。

### 4.5 Data 和 Buffer 结构体

`Data` 描述单个 token 的数据布局（字节数 + 是否需要 TMA 16 字节对齐）。`Buffer` 在 `Data` 基础上加了 `num_ranks` 和 `num_max_tokens_per_rank`，提供 `get_rank_buffer(rank_idx)` 和 `get_data_buffer(token_idx)` 方法来索引具体 token 的数据指针。kernel 中所有 buffer 都是在 workspace 末尾依次排列的 `Buffer` 实例，通过 `get_end_ptr()` 链式计算偏移。

## 5. JIT 编译和 Heuristic 配置

### 5.1 JIT 编译流程

`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` 中的 `sm100_fp8_fp4_mega_moe` 函数负责：

1. 调用 `get_mega_moe_config` 获取运行时配置。
2. 创建 10 个 TMA descriptor（见 5.3）。
3. 通过 `SM100FP8FP4MegaMoERuntime` 生成 C++ 模板实例化代码，JIT 编译。
4. 以 `grid_dim = num_sms`、`cluster_dim = 2`、动态 shared memory = `config.smem_size` 启动 kernel。

kernel 的 22 个模板参数全部在编译时确定，包括 shape、block config、stage 数、线程数等。

### 5.2 Heuristic 配置选择

`get_mega_moe_config` 是配置的总入口，依次调用三个子函数：

#### (a) Block 配置：`get_block_config_for_mega_moe`

根据 `expected_tokens_per_expert = num_tokens * num_ranks * num_topk / num_experts` 选择 BLOCK_M：

| expected tokens/expert | BLOCK_M | STORE_BLOCK_M | epilogue warpgroups | 典型场景 |
|---|---|---|---|---|
| ≤ 8.5 | 16 | 8 | 2 | RL 长尾 rollout |
| ≤ 16.5 | 32 | 16 | 2 | 小 batch, EP8, bsz 128 |
| ≤ 32.5 | 64 | 32 | 1 | 中 batch, EP8, bsz 256 |
| ≤ 64.5 | 96 | 16 | 2 | 大 batch, EP8, bsz 512 |
| ≤ 96.5 | 128 | 32 | 2 | 中 batch, EP16, bsz 256 |
| > 96.5 | 192 | 32 | 2 | Prefill 或大 EP decoding |

`BLOCK_N = 128`、`BLOCK_K = 128` 固定不变。`cluster_size` 始终为 2。

核心意图：小 batch/decoding 场景减少 M 维 padding 浪费，大 batch/prefill 场景用更大 BLOCK_M 提升 Tensor Core 利用率。

#### (b) Expert Wave 配置：`get_num_experts_per_wave_for_mega_moe`

```
// 伪代码
expected_tokens_per_expert = num_tokens * num_topk / num_experts_per_rank

if expected_tokens_per_expert < 1:
    return num_experts_per_rank  // 大部分 expert 没有 token，一波全做

// 估算每个 expert 的 L1 block 数
num_m_blocks = ceil(expected_tokens_per_expert / block_m)
num_n_blocks = (2 * intermediate_hidden) / block_n
num_l1_blocks_per_expert = num_m_blocks * num_n_blocks

// 乘以 imbalance_factor=2 来补偿路由不均匀
num_experts_per_wave = ceil(2 * num_sms / num_l1_blocks_per_expert)
num_experts_per_wave = min(num_experts_per_wave, num_experts_per_rank)

// 向上取整到 num_experts_per_rank 的因子，保证每波处理相同数量
while num_experts_per_wave < num_experts_per_rank
      and num_experts_per_rank % num_experts_per_wave != 0:
    num_experts_per_wave += 1
```

wave 的作用是控制 L1→L2 的切换粒度：一波内先做完所有 expert 的 L1，再做同一批 expert 的 L2，使得 L1 产生的 L2 输入有机会批量 ready。

#### (c) Pipeline 配置：`get_pipeline_config_for_mega_moe`

计算 shared memory 布局和最大 pipeline stage 数：

```
Shared Memory 布局:
┌─────────────────────────────────────────────────────────────┐
│ 固定区域 (smem_fixed):                                       │
│  ├─ dispatch: expert_count[num_experts] (align 1024)         │
│  ├─ dispatch: send_buffers[num_dispatch_warps] (align 1024)  │
│  ├─ C/D output: max(L1_FP8 × 2 stages, L2_BF16 × 1 stage)  │
│  ├─ amax reduction: store_block_m × num_epilogue_warps × 4B  │
│  ├─ barriers: dispatch + tmem_full/empty + combine           │
│  └─ tmem pointer: 4 bytes                                    │
├─────────────────────────────────────────────────────────────┤
│ Per-stage (× num_stages):                                    │
│  ├─ A tile: load_block_m × block_k × sizeof(FP8)            │
│  ├─ B tile: block_n × block_k × sizeof(FP4_unpacked)        │
│  ├─ SFA: sf_block_m × 4 bytes                               │
│  ├─ SFB: sf_block_n × 4 bytes                               │
│  └─ full/empty barriers: 2 × 8 bytes                        │
└─────────────────────────────────────────────────────────────┘

num_stages = (SM100_smem_capacity - smem_fixed) / smem_per_stage
// SM100 smem capacity = 232,448 bytes
// 要求 num_stages >= 2
```

### 5.3 TMA Descriptor 创建

kernel 使用 10 个 TMA descriptor，分为 L1 和 L2 两组：

| Descriptor | 用途 | Global Shape | Tile Shape | Swizzle |
|---|---|---|---|---|
| `tensor_map_l1_acts` | L1 输入 token | `[hidden, pool_tokens]` | `[block_k, load_block_m]` | 128B |
| `tensor_map_l1_acts_sf` | L1 输入 SF | `[padded_sf_tokens, hidden/128]` | `[sf_block_m, 1]` | 无 |
| `tensor_map_l1_weights` | L1 权重 | `[hidden, experts×inter_hidden×2]` | `[block_k, load_block_n]` | 128B |
| `tensor_map_l1_weights_sf` | L1 权重 SF | `[inter_hidden×2, hidden]` grouped | `[block_n, 1]` | 无 |
| `tensor_map_l1_output` | L1 输出(=L2输入) | `[inter_hidden, pool_tokens]` | `[block_n/2, store_block_m]` | 64B |
| `tensor_map_l2_acts` | L2 输入 token | `[inter_hidden, pool_tokens]` | `[block_k, load_block_m]` | 128B |
| `tensor_map_l2_acts_sf` | L2 输入 SF | `[padded_sf_tokens, inter_hidden/128]` | `[sf_block_m, 1]` | 无 |
| `tensor_map_l2_weights` | L2 权重 | `[inter_hidden, experts×hidden]` | `[block_k, load_block_n]` | 128B |
| `tensor_map_l2_weights_sf` | L2 权重 SF | `[hidden, inter_hidden]` grouped | `[block_n, 1]` | 无 |

注意 `tensor_map_l1_output` 的 swizzle 是 64B（128B 的一半），因为 SwiGLU 后输出宽度减半（`BLOCK_N/2`）。L1 output 和 L2 acts 实际指向同一块内存（`l2_acts`）。

L1 acts 的 TMA 使用 multicast（`load_block_m = block_m / 2`），两个 CTA 各加载一半 M 维度的 token。

## 6. Device Scheduler 状态机

`MegaMoEScheduler` 在 device 侧根据每个 local expert 的 token 数产生 GEMM block 序列。它是一个 persistent kernel 的核心调度器，每个 SM 独立运行一个实例。

### 6.1 初始化与 token 计数缓存

`fetch_expert_recv_count` 等待所有 expert 的 `expert_recv_count_sum` 就绪：

```cpp
// 每个 lane 缓存 expert (i * 32 + lane_idx) 的 token 数
for i in 0..kNumExpertsPerLane:
    expert_idx = i * 32 + lane_idx
    if expert_idx < kNumExpertsPerRank:
        // 自旋等待高 32 位 == kNumSMs * kNumRanks（所有 SM 和 rank 都贡献完）
        do { value = ld_volatile(expert_recv_count_sum[expert_idx]) }
        while (value >> 32) != kNumSMs * kNumRanks
        stored_num_tokens_per_expert[i] = (uint32_t)value  // 低 32 位是 token 数
```

这种 per-lane 缓存设计使得后续查询任意 expert 的 token 数只需一次 warp shuffle（`ptx::exchange`），查询 pool block offset 只需一次 warp reduction（`__reduce_add_sync`）。

### 6.2 Block 分配逻辑

每个 SM 从 `block_idx = blockIdx.x` 开始，每取到一个 block 后 `block_idx += kNumSMs`，这样整个 grid 上的 SM 共同遍历所有 expert 的 M×N blocks。

block 在 expert 内的映射：
```
// L1: 每个 expert 有 num_m_blocks × kNumL1BlockNs 个 block
m_block_idx = block_idx / kNumL1BlockNs
n_block_idx = block_idx % kNumL1BlockNs

// L2: 每个 expert 有 num_m_blocks × kNumL2BlockNs 个 block
m_block_idx = block_idx / kNumL2BlockNs
n_block_idx = block_idx % kNumL2BlockNs
```

当一个 expert 的所有 block 分配完后，`block_idx` 减去已消耗的 block 数，继续分配下一个 expert。

### 6.3 Wave 状态机

调度状态机有两个 phase，按 wave 切换：

```
                    ┌──────────────────────────────────────────┐
                    │         Wave 0 (expert 0..W-1)           │
                    │  ┌─────────┐         ┌─────────┐        │
                    │  │ Linear1 │ ──完──→ │ Linear2 │        │
                    │  │ (L1)    │         │ (L2)    │        │
                    │  └─────────┘         └─────────┘        │
                    └──────────────────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │         Wave 1 (expert W..2W-1)          │
                    │  ┌─────────┐         ┌─────────┐        │
                    │  │ Linear1 │ ──完──→ │ Linear2 │        │
                    │  │ (L1)    │         │ (L2)    │        │
                    │  └─────────┘         └─────────┘        │
                    └──────────────────────────────────────────┘
                                       │
                                       ▼
                                     ......
```

`get_next_block` 的核心逻辑（伪代码）：

```
while current_local_expert_idx < kNumExpertsPerRank:
    if next_phase == Linear1:
        if fetch_next_l1_block():  // 在当前 wave 内找到 L1 block
            n_block_idx = block_idx - m_block_idx * kNumL1BlockNs
            block_idx += kNumSMs
            return (Linear1, expert_idx, m_block_idx, n_block_idx)
        else:
            // 当前 wave 的 L1 全部分配完，切到 L2
            next_phase = Linear2
            // 回退到当前 wave 的起始 expert（向下对齐到 wave 边界）
            set_expert_idx(align_down(current_local_expert_idx - 1, kNumExpertsPerWave))
    else:  // Linear2
        if fetch_next_l2_block():
            n_block_idx = block_idx - m_block_idx * kNumL2BlockNs
            block_idx += kNumSMs
            return (Linear2, expert_idx, m_block_idx, n_block_idx)
        else:
            // 当前 wave 的 L2 全部分配完，进入下一波的 L1
            next_phase = Linear1

return (None, 0, 0, 0)  // 所有 wave 处理完毕
```

L2 的调度不会立刻跟在每个 L1 block 后，而是按 expert wave 切换。这使得 L1 产生的 L2 输入有机会批量 ready，也降低 L1/L2 之间细粒度等待的开销。

### 6.4 Cluster 约束

scheduler 要求 `kNumL1BlockNs` 和 `kNumL2BlockNs` 都是偶数，`kNumSMs` 也是偶数。这保证 cluster 中的 2 个 CTA 总是落在同一个 `m_block_idx` 上，`n_block_idx` 相差 1，从而可以共享 multicast 的 A tile。

## 7. 核心 Kernel 的线程角色划分

`sm100_fp8_fp4_mega_moe_impl` 是一个 cluster size 2 的 persistent kernel。

### 7.1 线程布局

总线程数 = `kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads`。

```
线程角色分配 (以 warp_idx 划分):
┌──────────────────────────────────────────────────────────────────────┐
│ warp 0..3 (kNumDispatchWarps=4, 128 threads, 48 regs/thread)        │
│   Dispatch warps:                                                    │
│   - 统计 expert token 数                                             │
│   - 写跨 rank source index                                          │
│   - 拉取远端 token 到本地 L1 pool                                    │
│   - 清理 workspace                                                   │
├──────────────────────────────────────────────────────────────────────┤
│ warp 4 (kNumDispatchWarps+0, 40 regs/thread)                        │
│   TMA load warp: acts + SFA                                         │
│   - L1: tensor_map_l1_acts + tensor_map_l1_acts_sf                  │
│   - L2: tensor_map_l2_acts + tensor_map_l2_acts_sf                  │
│   - 等待 l1_arrival_count / l2_arrival_mask                         │
├──────────────────────────────────────────────────────────────────────┤
│ warp 5 (kNumDispatchWarps+1, 40 regs/thread)                        │
│   TMA load warp: weights + SFB                                      │
│   - L1: tensor_map_l1_weights + tensor_map_l1_weights_sf            │
│   - L2: tensor_map_l2_weights + tensor_map_l2_weights_sf            │
├──────────────────────────────────────────────────────────────────────┤
│ warp 6 (kNumDispatchWarps+2, 40 regs/thread)                        │
│   UMMA issue warp (仅 leader CTA 执行)                               │
│   - UTCCP copy SFA/SFB 到 tensor memory                             │
│   - 发射 SM100_MMA_MXF8F6F4_2x1SM_SS 指令                          │
│   - 管理 accumulator 双缓冲                                          │
├──────────────────────────────────────────────────────────────────────┤
│ warp 7 (kNumDispatchWarps+3, 40 regs/thread)                        │
│   空闲 warp (non-epilogue 凑满 128 threads)                          │
├──────────────────────────────────────────────────────────────────────┤
│ warp 8..15 (epilogue, 208 regs/thread, 1-2 warpgroups)              │
│   Epilogue warps:                                                    │
│   - L1 epilogue: SwiGLU + topk_weight + FP8 量化 + TMA store        │
│   - L2 epilogue: BF16 cast + NVLink write-back to combine buffer    │
│   - Final combine: reduce top-k slots → TMA store to y              │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 寄存器分配

kernel 使用 `warpgroup_reg_dealloc` / `warpgroup_reg_alloc` 做异构寄存器分配：

- Dispatch warps: 48 regs/thread（轻量级，主要做地址计算和 NVLink 读写）
- Non-epilogue warps: 40 regs/thread（TMA load 和 UMMA issue 不需要太多寄存器）
- Epilogue warps: 208 regs/thread（需要大量寄存器做 SwiGLU 计算、amax reduction、FP8 cast）

总寄存器约束：`48×128 + 40×128 + 208×256 = 64,512 ≤ 64,512`（SM100 上限）。

### 7.3 初始化阶段

kernel 开始时按 warp 分工初始化：

```
warp 0: 清零 shared memory 中的 smem_expert_count
warp 1: 初始化 dispatch barriers (kNumDispatchWarps 个, 各 arrive count = 1)
warp 2: 初始化 GEMM barriers
         - full_barriers[i]: arrive count = 2×2 (2 CTA × 2 producers)
         - empty_barriers[i]: arrive count = 1
         - tmem_full_barriers[i]: arrive count = 1
         - tmem_empty_barriers[i]: arrive count = 2 × kNumEpilogueThreads
         - combine_barriers[i]: arrive count = 1
warp 3: 分配 tensor memory (Allocator2Sm, kNumTmemCols 列)
```

两次 `cluster_sync_with_relaxed_arrive` 确保：
1. 第一次：TMA descriptor prefetch 完成。
2. 第二次：barrier 初始化和 tensor memory 分配完成，两个 CTA 状态一致。

## 8. Dispatch 阶段详解

dispatch 线程（warp 0-3）负责整个跨 rank token 路由过程，分为以下步骤。

### 8.1 遍历 topk_idx 的通用模式

dispatch 使用一个 `read_topk_idx` 闭包来遍历本 rank 的所有 token-topk 对：

```cpp
// 每个 warp 一次处理 kNumTokensPerWarp = 32 / num_topk 个 token
// 一个 warp 的 32 个 lane 覆盖这些 token 的所有 topk slots
// 例如 num_topk=6 时，每个 warp 处理 5 个 token，使用 30 个 lane
for (i = (sm_idx * 4 + warp_idx) * kNumTokensPerWarp;
     i < num_tokens;
     i += kNumSMs * 4 * kNumTokensPerWarp):
    if i + lane_idx/kNumTopk < num_tokens and lane_idx < kNumActivateLanes:
        expert_idx = topk_idx[i * kNumTopk + lane_idx]
        if expert_idx >= 0:
            process(i * kNumTopk + lane_idx, expert_idx)
    __syncwarp()
```

### 8.2 统计 expert token 数

第一次遍历 `topk_idx`，对每个有效 expert 在 block shared memory 的 `smem_expert_count[expert_idx]` 做 `atomicAdd_block`：

```
read_topk_idx(lambda (token_topk_idx, expert_idx):
    atomicAdd_block(smem_expert_count[expert_idx], 1)
)
// barrier: 所有 dispatch threads 同步
```

### 8.3 生成全局发送 offset

每个 SM 将本 SM 统计到的 expert count 用 `atomic_add` 加到 workspace 的 `expert_send_count`：

```
for i in thread_idx..kNumExperts step kNumDispatchThreads:
    // send_value: 高32位=1(完成标记), 低32位=本SM的token数
    send_value = (1ULL << 32) | smem_expert_count[i]
    old_value = atomic_add(expert_send_count[i], send_value)
    // old_value 的低32位 = 之前所有SM累计的token数 = 本SM的slot起始offset
    smem_expert_count[i] = (uint32_t)old_value
// barrier
```

### 8.4 写远端 source index

第二次遍历 `topk_idx`。对每个 token-topk：

```
read_topk_idx(lambda (token_topk_idx, expert_idx):
    dst_rank_idx = expert_idx / kNumExpertsPerRank
    dst_local_expert = expert_idx % kNumExpertsPerRank
    dst_slot_idx = atomicAdd_block(smem_expert_count[expert_idx], 1)
    // 通过 NVLink 写到目标 rank 的 workspace
    *sym_buffer.map(
        workspace.src_token_topk_idx[dst_local_expert][my_rank][dst_slot_idx],
        dst_rank_idx
    ) = token_topk_idx
)
```

这一步只写 source index（`token_topk_idx = token_idx * num_topk + topk_slot`），还没有搬运 token 数据。

### 8.5 Grid sync + 汇总 expert 接收数

Grid sync 后，仅 SM0 将 `expert_send_count` 转写为各目标 rank 的接收计数：

```
if sm_idx == 0:
    for i in thread_idx..kNumExperts step kNumDispatchThreads:
        dst_rank = i / kNumExpertsPerRank
        dst_expert = i % kNumExpertsPerRank
        status = expert_send_count[i]  // 低32位=token数, 高32位=完成SM数

        // 写到目标 rank 的 expert_recv_count（普通写）
        *sym_buffer.map(expert_recv_count[my_rank, dst_expert], dst_rank) = status & 0xFFFFFFFF

        // 写到目标 rank 的 expert_recv_count_sum（system-scope atomic add）
        // 高32位累计完成状态，低32位累加token数
        atomic_add_sys(sym_buffer.map(expert_recv_count_sum[dst_expert], dst_rank), status)
```

### 8.6 NVLink barrier → Pull 阶段

NVLink barrier 确保所有 rank 的 source index 和 expert count 都已写完。然后进入 pull 阶段。

### 8.7 拉取远端 token 到本地 L1 pool

每个 dispatch warp 独立遍历本 rank 所有 local expert 的 token pool：

```
// 缓存 expert token 数（复用 scheduler 的 fetch_expert_recv_count）
scheduler.fetch_expert_recv_count()

for token_idx = sm_idx*4+warp_idx; ; token_idx += kNumSMs*4:
    // 推进到包含 token_idx 的 expert
    while token_idx >= expert_end_idx:
        current_expert_idx++
        expert_end_idx += scheduler.get_num_tokens(current_expert_idx)

    // Round-robin min-peeling 确定源 rank
    // (见 8.8 详解)
    (src_rank, token_idx_in_rank) = round_robin_select(token_idx_in_expert)

    // 读目标 workspace 中的 src_token_topk_idx
    src_token_topk_idx = workspace.src_token_topk_idx[expert][src_rank][token_idx_in_rank]
    src_token_idx = src_token_topk_idx / kNumTopk
    src_topk_idx  = src_token_topk_idx % kNumTopk

    // 1. TMA load: 从源 rank 拉 FP8 token 到 shared memory
    tma_load_1d(pull_buffer, sym_buffer.map(input_token[src_token_idx], src_rank))

    // 2. 直接加载源 rank 的 SF，按 UTCCP 变换后写到本地 l1_sf_buffer
    for each sf_uint32:
        l1_sf[sf_k * stride + transform_sf_token_idx(token_in_expert)] = remote_sf[sf_k]

    // 3. 读源 rank 的 topk_weight，写到 l1_topk_weights[pool_token_idx]
    // 4. 等 TMA load 完成，TMA store token 到 l1_token[pool_token_idx]
    // 5. 写 TokenSrcMetadata{src_rank, src_token_idx, src_topk_idx}
    // 6. release add l1_arrival_count[pool_block_idx] += 1
```

### 8.8 Round-robin min-peeling 算法

pull 阶段需要根据 `token_idx_in_expert`（该 expert 内的第几个 token）反推出它来自哪个源 rank。代码使用 min-peeling 而非简单的按 rank 顺序拼接：

```
// 每个 lane 缓存一个 rank 的 remaining token 数
remaining[i] = stored_rank_count[i]  // 各 rank 发给该 expert 的 token 数
offset = 0
slot_idx = token_idx_in_expert

while true:
    // warp reduce: 活跃 rank 数和最小 remaining
    num_active_ranks = warp_reduce_add(remaining[i] > 0)
    min_length = warp_reduce_min(remaining[i])

    // 本轮 token 数 = min_length × 活跃 rank 数
    num_round_tokens = min_length * num_active_ranks
    if slot_idx < num_round_tokens:
        // 命中本轮：slot_idx % num_active_ranks 确定是第几个活跃 rank
        // 用 __ballot_sync + __fns 找到对应的实际 rank_idx
        src_rank = find_nth_active_rank(slot_idx % num_active_ranks)
        token_in_rank = offset + slot_idx / num_active_ranks
        break

    // 未命中，进入下一轮
    slot_idx -= num_round_tokens
    offset += min_length
    remaining[i] -= min(remaining[i], min_length)
```

这种方式不是简单地按 rank 0 全部、rank 1 全部的顺序拉取，而是按各 rank 剩余 token 数轮转，使跨 rank 的 NVLink 访问更均匀，避免某个 rank 的 NVLink 带宽成为瓶颈。

## 9. L1/L2 GEMM Pipeline

MMA pipeline 由三个 non-epilogue warp 角色组成，各自独立运行 `scheduler.for_each_block` 循环。

### 9.1 Pipeline 总览

```
Acts TMA warp (warp 4)          Weights TMA warp (warp 5)       UMMA warp (warp 6, leader CTA only)
      │                                │                                │
      ▼                                ▼                                ▼
┌─────────────┐                ┌─────────────┐                  ┌──────────────┐
│ 等待 arrival │                │             │                  │ 等待 tmem    │
│ count/mask   │                │             │                  │ empty barrier│
├─────────────┤                ├─────────────┤                  ├──────────────┤
│ for each K: │                │ for each K: │                  │ for each K:  │
│  wait empty │                │  wait empty │                  │  wait full   │
│  TMA load A │                │  TMA load B │                  │  UTCCP SF→TM │
│  TMA load SF│                │  TMA load SF│                  │  issue UMMA  │
│  arrive full│                │  arrive full│                  │  arrive empty│
└─────────────┘                └─────────────┘                  │  (last K:    │
                                                                │   arrive tmem│
                                                                │   full)      │
                                                                └──────────────┘
                                                                        │
                                                                        ▼
                                                                Epilogue warps
                                                                (wait tmem full)
```

### 9.2 L1 等待 token arrival

Acts TMA load warp 在处理 L1 block 前会自旋等待：

```cpp
// L1: 等待 dispatch 把当前 M block 内所有有效 token 写到 L1 pool
while (ld_acq(l1_arrival_count[pool_block_idx]) != valid_m);
```

### 9.3 L2 等待 L1 output arrival

处理 L2 block 前等待 `l2_arrival_mask`：

```cpp
// L2: 等待所有 L1 N blocks 的 epilogue 完成
// 每个 L1 N block 完成后设置 1 bit，共 2*num_k_blocks 个 bit
// (因为 BLOCK_K == BLOCK_N，L1 的 BLOCK_N/2 对应 L2 的一个 K block，
//  所以每个 L1 N block 贡献 1 bit，总共 kNumL1BlockNs 个 bit)
expected = ((1ULL << num_k_blocks) << num_k_blocks) - 1;
while (ld_acq_gpu(l2_arrival_mask[pool_block_idx]) != expected);
```

注意：代码注释提到，原本设计是按需等待单个 K block 以重叠 L1 计算和 L2 加载，但实测发现当 `num_experts_per_wave` 足够大时（保证 L1 在 L2 开始前已完成），这种优化反而是负面的，所以当前实现是一次性等待所有 bit。

### 9.4 TMA Load

Acts warp 和 Weights warp 根据当前 `block_phase` 选择对应的 TMA descriptor：

```
Acts warp:
  L1 → tensor_map_l1_acts + tensor_map_l1_acts_sf
  L2 → tensor_map_l2_acts + tensor_map_l2_acts_sf

Weights warp:
  L1 → tensor_map_l1_weights + tensor_map_l1_weights_sf
  L2 → tensor_map_l2_weights + tensor_map_l2_weights_sf
```

每个 K block 的 TMA load 流程：
1. 等待 `empty_barriers[stage_idx]`（consumer 释放 stage）。
2. TMA copy A/B tile 到 shared memory（multicast 到 cluster 内 2 个 CTA）。
3. TMA copy SFA/SFB 到 shared memory。
4. Leader CTA: `arrive_and_expect_tx` 设置预期字节数。Non-leader CTA: `arrive(0)`。

Acts warp 的 multicast：非 leader CTA 的 `m_idx` 会偏移 `valid_m/2`，使两个 CTA 各加载 M 维的一半（`LOAD_BLOCK_M = BLOCK_M / 2`）。

### 9.5 UMMA Issue

UMMA warp 只在 leader CTA 上运行（`is_leader_cta`）。

```
MMA 配置:
  - A/B swap: weights 作为 A (FP4 E2M1), acts 作为 B (FP8 E4M3)
  - 指令: SM100_MMA_MXF8F6F4_2x1SM_SS (2-CTA shared-shared)
  - UMMA_M = 256 (= LAYOUT_AD_M * 2 = 128 * 2)
  - UMMA_N = BLOCK_M (swap 后 M 变成 N)
  - UMMA_K = 32
  - Accumulator: tensor memory, 双 stage (kNumEpilogueStages = 2)
```

每个 K block 的 UMMA 流程：
1. 等待 `full_barriers[stage_idx]`（TMA load 完成）。
2. UTCCP copy SFA/SFB 从 shared memory 到 tensor memory。
3. 发射 `BLOCK_K / UMMA_K = 128 / 32 = 4` 次 UMMA 指令。
4. `empty_barrier_arrive`：释放 TMA stage。
5. 最后一个 K block 还会 signal `tmem_full_barriers`，通知 epilogue 可以消费 accumulator。

UMMA N 维度会根据实际有效 M 动态调整（`update_instr_desc_with_umma_n`），避免对 padding 行做无用计算。

## 10. L1 Epilogue：SwiGLU、top-k weight、FP8 量化

当 scheduler 当前 block phase 是 `Linear1`，epilogue warps 执行以下流程。

### 10.1 Epilogue 分块结构

epilogue 将 BLOCK_M 分成多层：

```
BLOCK_M
├── WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups  (每个 warpgroup 负责的 M 范围)
│   ├── STORE_BLOCK_M  (每次 TMA store 的 M 范围)
│   │   ├── ATOM_M = 8  (每次 TMEM load 的最小 M 单元)
│   │   └── ...
│   └── ...
└── ...
```

### 10.2 SwiGLU 计算详解

对每个 ATOM_M = 8 的块：

```
// 1. 从 tensor memory 加载 accumulator (FP32)
//    SM100_TMEM_LOAD_16dp256b1x 指令一次加载 8 个 FP32 值
//    由于 L1 权重做了 gate/up interleave (granularity 8)，
//    硬件排列使得 gate/up 对为:
//      (values[0], values[2]), (values[1], values[3])  — 上半
//      (values[4], values[6]), (values[5], values[7])  — 下半

// 2. 每 32 个 token 从 l1_topk_weights_buffer 加载一次 weight 到寄存器缓存
//    通过 warp shuffle (ptx::exchange) 广播到需要的 lane

// 3. 对每对 (gate, up):
gate_bf16 = float_to_bf16(values[k*4], values[k*4+1])
up_bf16   = float_to_bf16(values[k*4+2], values[k*4+3])

// 可选 clamp (当 kActivationClamp != inf)
if clamp:
    gate_bf16 = min(gate_bf16, clamp)
    up_bf16 = clamp(up_bf16, -clamp, clamp)

// SwiGLU = silu(gate) * up = gate / (1 + exp(-gate)) * up
gate_f32 = bf16_to_float(gate_bf16)
neg_gate_exp = exp(-gate_f32)          // 可选 fast_math: __expf
denom = 1.0 + neg_gate_exp
silu_gate = gate_f32 / denom           // 可选 fast_math: fast_rcp
up_f32 = bf16_to_float(up_bf16)
swiglu = silu_gate * up_f32 * topk_weight

// 4. Amax reduction: 4 个 lane 一组做 warp reduce max
//    写到 smem_amax_reduction 供跨 warp 交换
```

### 10.3 FP8 量化和 TMA Store

```
// 5. 跨 warp 交换 amax（两个 warp 共享同一组 M 行）
//    取两个 warp 的 amax 最大值

// 6. 计算 FP8 E4M3 的 scale factor 和 inverse scale
(sf, sf_inv) = get_e4m3_sf_and_sf_inv(amax)

// 7. 乘 sf_inv 后 cast 成 FP8 E4M3
fp8_values = cast_to_fp8_e4m3(swiglu * sf_inv)

// 8. STSM: 写到 shared memory (swizzled 布局)
//    使用 SM100_U8x4_STSM_T 指令

// 9. Warpgroup barrier 同步

// 10. TMA store 到 l2_acts (= L2 输入)
//     使用 tensor_map_l1_output descriptor
//     注意: L1 output 宽度是 BLOCK_N/2 (SwiGLU 将 gate+up 合并)

// 11. 写 SF 到 l2_sf_buffer (UE8M0 格式, MN-major UTCCP 布局)
//     只有 warp_idx_in_wg % 2 == 0 且 lane_idx < 4 的线程写 SF
//     SF 地址计算使用优化后的 transform_sf_token_idx
```

### 10.4 通知 L2

```
// 12. 等待 TMA store 完成
tma_store_wait<0>()

// 13. Epilogue 全局同步
sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx)

// 14. 设置 l2_arrival_mask 的对应 bit
if epilogue_warp_idx == 0:
    red_or_rel_gpu(l2_arrival_mask[pool_block_idx], 1ULL << n_block_idx)
```

这里 L1 的输出没有落 BF16 中间张量，而是直接量化为 L2 GEMM 所需的 FP8 输入格式，避免了一次额外的全局内存读写。

## 11. L2 Epilogue：写回远端 Combine Buffer

当 scheduler 当前 block phase 是 `Linear2`，epilogue warps 执行以下流程。

### 11.1 TMEM Load → BF16 Cast → Shared Memory

```
for each store_block (s in 0..WG_BLOCK_M/STORE_BLOCK_M):
    for each atom (i in 0..STORE_BLOCK_M/ATOM_M):
        // 从 tensor memory 加载 FP32 accumulator
        SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, values[0..7])

        // 最后一个 atom 时 signal tmem_empty_barriers（释放 accumulator）

        // Cast FP32 → BF16 并打包
        packed = cast_into_bf16_and_pack(values[0], values[1], ...)

        // STSM 写到 shared memory (BF16 swizzled 布局)
        // 2 个 warp 共享一个 BF16 swizzle atom
        SM90_U32x4_STSM_T::copy(packed, smem_ptr)
```

### 11.2 NVLink Write-back

```
    // Warpgroup barrier: 等待 shared memory 写入完成
    sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx)

    // 每个 warp 负责若干行，每行独立写回
    for each row (j in 0..kNumRowsPerWarp):
        row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16
        if row >= valid_m: break  // 跳过 padding 行

        // 读 TokenSrcMetadata: 这个 pool token 要写回哪里
        metadata = workspace.token_src_metadata[m_idx + row]
        dst_rank = metadata.rank_idx
        dst_token = metadata.token_idx
        dst_topk = metadata.topk_idx

        // 从 shared memory 读 BF16 数据 (16 bytes per lane)
        packed = ld_shared<float4>(smem_ptr)

        // 通过 NVLink 写到目标 rank 的 combine buffer
        dst_ptr = combine_token_buffer[dst_topk][dst_token] + n_offset
        *sym_buffer.map(dst_ptr, dst_rank) = packed
```

L2 输出在专家所在 rank 计算，但结果直接通过 symmetric memory 写回 token 原始 rank 的 combine buffer。这个阶段相当于 fused EP combine 的 write-back 部分。

与 L1 epilogue 的关键区别：
- L2 不做量化，直接输出 BF16。
- L2 不用 TMA store，而是通过 NVLink 直接写远端内存（每行目标 rank 可能不同）。
- L2 没有 amax reduction 和 SF 计算。

## 12. Final Combine Reduce

所有 L2 输出写入 combine buffer 后，epilogue warps 先释放 tensor memory，然后进入 NVLink barrier 确保所有 rank 的写回都完成。接着与 dispatch warps 做一次 barrier 同步（`kDispatchWithEpilogueBarrierIdx`），让 dispatch warps 可以开始清理 workspace。

### 12.1 Chunk 策略

combine reduce 复用 GEMM 阶段的 shared memory（barrier 之前的区域）。为了平衡 shared memory 和寄存器压力，hidden 维度可能被分成 1 或 2 个 chunk：

```cpp
// 3 个 slot: 2 个 load stage + 1 个 store
kNumChunkSlots = 3;

// 选择 chunk 数: 同时满足 smem 和寄存器约束
kNumChunks =
    (3 * kNumEpilogueWarps * kHidden * 2 <= SMEM_BEFORE_BARRIER_SIZE
     && kHidden <= 32 * 128) ? 1 : 2;

kNumChunkBytes = kHidden * sizeof(BF16) / kNumChunks;
```

### 12.2 双缓冲 TMA Load + Reduce

每个 epilogue warp 独立处理一个 token：

```
for token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
    token_idx < num_tokens;
    token_idx += kNumSMs * kNumEpilogueWarps:

    // 读 topk_idx，每个 lane 读一个 slot
    stored_topk_slot = topk_idx[token_idx * kNumTopk + lane_idx]  // lane < kNumTopk
    total_mask = __ballot_sync(stored_topk_slot >= 0)  // 有效 slot 的 bitmask

    for each chunk:
        // 双缓冲: load stage 0/1 交替
        // 先加载第一个有效 topk slot
        slot_idx = __ffs(mask) - 1
        TMA load combine_buffer[slot_idx][token_idx][chunk] → load_buffer[0]

        // 累加循环
        float2 reduced[...] = {0}
        while has_more_slots:
            // 预取下一个 slot 到另一个 buffer
            next_slot = __ffs(remaining_mask) - 1
            TMA load combine_buffer[next_slot][token_idx][chunk] → load_buffer[1]

            // 等待当前 buffer ready，累加 BF16 → FP32
            wait(combine_load_barriers[current_stage])
            for each uint4 element:
                reduced[j] += bf16_to_float2(load_buffer[current_stage][j])

            // 交换 stage
            swap(current_stage)

        // Cast FP32 → BF16，写到 store buffer
        for each element:
            store_buffer[j] = float2_to_bf16(reduced[j])

        // TMA store 到最终输出 y[token_idx]
        tma_store_1d(y + token_idx * hidden_bytes + chunk_offset, store_buffer, chunk_bytes)
```

注意：top-k weight 已经在 L1 epilogue 乘过，所以 final combine 只是加和各 top-k 分支的 BF16 输出。

## 13. 同步机制总览

Mega MoE 同时用了多层同步，下面按层级从低到高梳理。

### 13.1 Intra-warp / Intra-warpgroup 同步

- `__syncwarp()`: warp 内 32 线程同步。
- `ptx::sync_aligned(N, barrier_idx)`: N 个线程的 named barrier 同步（要求线程 ID 对齐）。
- `ptx::sync_unaligned(N, barrier_idx)`: N 个线程的 named barrier 同步（线程 ID 不要求对齐，用于 dispatch + epilogue 跨角色同步）。

kernel 中使用的 named barrier：

| Barrier Index | 参与线程 | 用途 |
|---|---|---|
| `kDispatchBarrierIdx = 0` | dispatch threads (128) | dispatch 各步骤间同步 |
| `kDispatchWithEpilogueBarrierIdx = 1` | dispatch + epilogue | dispatch 和 epilogue 的交接点 |
| `kEpilogueFullBarrierIdx = 2` | epilogue threads | L1 epilogue TMA store 完成后同步 |
| `kEpilogueWGBarrierStartIdx = 3+` | 128 per warpgroup | warpgroup 内 shared memory 读写同步 |

### 13.2 Cluster Barrier

kernel 使用 cluster size 2。初始化阶段有两次 `cluster_sync_with_relaxed_arrive`：
1. 第一次在 TMA descriptor prefetch 后，确保两个 CTA 的 prefetch 完成。
2. 第二次在 barrier 初始化和 tensor memory 分配后，确保两个 CTA 的 shared/tensor memory 状态一致。

使用 `.relaxed` arrive 是安全的，因为 `fence_barrier_init` 是 `.release.cluster`，而 `barrier.cluster.wait.aligned` 默认是 `.acquire`。

### 13.3 TMA / mbarrier Pipeline

TMA load/store 使用 `ClusterTransactionBarrier` 实现生产者-消费者流水线：

```
Pipeline 同步结构:

TMA Load (producer)                    UMMA (consumer/producer)              Epilogue (consumer)
     │                                        │                                    │
     │  ── full_barriers[stage] ──→           │                                    │
     │                                        │  ── tmem_full_barriers[accum] ──→  │
     │  ←── empty_barriers[stage] ──          │                                    │
     │                                        │  ←── tmem_empty_barriers[accum] ── │
     │                                        │                                    │
```

| Barrier | 方向 | 用途 |
|---|---|---|
| `full_barriers[kNumStages]` | TMA → UMMA | TMA load 完成，A/B/SF 在 shared memory ready |
| `empty_barriers[kNumStages]` | UMMA → TMA | UMMA 消费完 stage，TMA 可以覆写 |
| `tmem_full_barriers[2]` | UMMA → Epilogue | 一个 block 的所有 K 累加完成，accumulator ready |
| `tmem_empty_barriers[2]` | Epilogue → UMMA | Epilogue 消费完 accumulator，UMMA 可以覆写 |
| `combine_barriers[2×warps]` | TMA → Epilogue | Combine 阶段的 TMA load 双缓冲 |

### 13.4 Grid Sync

`comm::grid_sync` 在一个 rank 内跨所有 SM 同步。使用 workspace 中的 `grid_sync_count` 计数器和高位翻转 tag，实现类似 cooperative groups grid sync 的效果。

dispatch 和 epilogue 使用不同的 grid sync index（`kDispatchGridSyncIndex = 0`、`kEpilogueGridSyncIndex = 1`），避免两组线程的 grid sync 互相干扰。

### 13.5 NVLink Barrier

`comm::nvlink_barrier` 通过 symmetric memory 中的 signal 对所有 rank 做 cross-rank barrier。只有 SM0 参与跨 rank signal，其他 SM 通过前后的 grid sync 被带入同一个全局阶段。

kernel 中有三个跨 rank barrier 点：

| Tag | 时机 | 作用 |
|---|---|---|
| `kBeforeDispatchPullBarrierTag = 1` | dispatch 写完 source index 和 expert count 后 | 确保目标 rank 可以安全开始 pull |
| `kBeforeCombineReduceBarrierTag = 2` | 所有 L2 epilogue 写回完成后 | 确保原 rank 可以安全 reduce combine buffer |
| `kAfterWorkspaceCleanBarrierTag = 3` | workspace 清理完成后 | 确保所有 rank 清理完成，buffer 可安全复用 |

## 14. Workspace 清理和统计

dispatch warp 在 pull 结束后，通过 `kDispatchWithEpilogueBarrierIdx` 与 epilogue warp 同步（确保 epilogue 已经过了 NVLink barrier），然后开始清理 workspace。这个清理过程与 combine reduce 在时间上重叠。

```
清理分工:
┌─────────────────────────────────────────────────────────────┐
│ SM 0:                                                        │
│   清零 expert_send_count[0..num_experts]                     │
├─────────────────────────────────────────────────────────────┤
│ SM 1..kNumSMs-1 (按 expert 分摊):                            │
│   for expert_idx = sm_idx-1; expert_idx < num_experts_per_rank; │
│       expert_idx += kNumSMs-1:                               │
│                                                              │
│     // 读取 token 数（清零前）                                │
│     num_recv_tokens = expert_recv_count_sum[expert_idx]      │
│                                                              │
│     // warp 0: 清零 expert_recv_count_sum                    │
│     // warp 1: 累加 cumulative_local_expert_recv_stats       │
│     //         (如果传入了该参数)                              │
│                                                              │
│     // 所有 warp: 清零 per-rank expert_recv_count            │
│     // 所有 warp: 清零 l1_arrival_count, l2_arrival_mask     │
└─────────────────────────────────────────────────────────────┘
```

清理完成后，通过 `kAfterWorkspaceCleanBarrierTag` NVLink barrier 确保所有 rank 清理完成，buffer 可以安全复用于下一次 kernel 调用。

## 15. 测试逻辑

`tests/test_mega_moe.py` 的测试流程：

1. 初始化 distributed group（`init_dist`）。
2. 分配 Mega MoE symmetric buffer（`get_symm_buffer_for_mega_moe`）。
3. 随机生成 BF16 输入、BF16 权重、top-k 路由（`torch.topk`）。
4. 可选地对 `topk_idx` 做随机 mask（`masked_ratio`），模拟无效路由。
5. 输入用 `per_token_cast_to_fp8(..., use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)` 转 FP8。
6. 权重按 expert 分组转 FP4（`per_token_cast_to_fp4`），SF 转成 DeepGEMM 要求的 MN-major 布局。
7. 调用 `transform_weights_for_mega_moe` 做 gate/up interleave 和 SF UTCCP 转置。
8. `run_fused` 每次先把输入 copy 到 buffer，再调用 `fp8_fp4_mega_moe`。
9. 如果 legacy 依赖可用（DeepEP + TileLang），用分阶段 baseline 做对比，要求 fused 与 baseline bitwise equal。
10. benchmark 输出 fused 时间、估算 TFLOPS、HBM GB/s、NVLink GB/s，以及相对 legacy 的加速比。

## 16. 关键限制和假设

- 当前 C++ dispatch 只支持 `arch_major == 10`（Blackwell SM100），其他架构会报 unsupported。
- 只支持 FP8 E4M3 activations + FP4 E2M1 weights。
- 只支持 `recipe=(1,1,32)`。
- 只支持 `activation=”swiglu”`。
- 依赖 PyTorch symmetric memory，README 标注需要 PyTorch >= 2.9。
- expert 数必须能被 rank 数整除（`num_experts % num_ranks == 0`）。
- `num_topk <= 32`，因为一个 warp 内用 lane mask 管 top-k slots。
- `hidden` 和 `intermediate_hidden` 必须是 128 的倍数（SF 和 TMA 对齐约束）。
- `kNumL1BlockNs` 和 `kNumL2BlockNs` 必须是偶数（cluster size 2 约束）。
- `num_experts_per_rank % num_experts_per_wave == 0`（wave 均匀划分约束）。
- kernel 假设 symmetric buffer 可在所有 rank 间通过 `sym_buffer.map` 访问，跨 rank 同步依赖 NVLink/system-scope atomic。
- 寄存器总量约束：`48×128 + 40×128 + 208×epilogue_threads ≤ 64,512`。

## 17. 一句话数据流

Mega MoE 的完整数据流可以概括为：

```
原 rank 输入 x/topk
  → 目标 expert rank 写 source index (NVLink)
  → 目标 rank 按 local expert pull 远端 token 到 L1 pool (NVLink + TMA)
  → L1 FP8×FP4 GEMM (UMMA, tensor memory accumulator)
  → L1 epilogue: SwiGLU × topk_weight → FP8 量化 → L2 acts (TMA store)
  → L2 FP8×FP4 GEMM (UMMA, tensor memory accumulator)
  → L2 epilogue: BF16 cast → 写回原 rank combine buffer (NVLink)
  → 原 rank 本地 reduce top-k slots (TMA load + FP32 累加)
  → TMA store → 输出 y
```

它的核心不是”减少了一次 Python 调用”，而是把 MoE 路由通信、中间激活生成、两次 GEMM 和最终 combine 放进同一个 persistent kernel，用 shared memory、tensor memory、TMA、UMMA、workspace counters 和 NVLink barrier 组织成跨 rank 的流水线。通信和计算在时间上重叠：dispatch pull 与 MMA pipeline 并行，workspace 清理与 combine reduce 并行。

