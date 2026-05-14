# 09 · Shared Memory 预算与 Pipeline 深度重算

> **难点强度：★★★☆☆**

## 1. SM100 shared memory 账本

SM100 SM smem 上限 232,448 B，Mega MoE 在 [jit_kernels/heuristics/mega_moe.hpp](csrc/jit_kernels/heuristics/mega_moe.hpp) 的 `get_pipeline_config_for_mega_moe` 中按 [mega_moe_code_analysis.md §5.2(c)](reports/mega_moe_code_analysis.md) 列出的两段做分配：

```
固定区 smem_fixed:
  - dispatch smem_expert_count[num_experts] (align 1024)
  - dispatch send_buffers[4 warps] (align 1024)
  - C/D output buffer = max(L1 FP8 × 2 stages, L2 BF16 × 1 stage)
  - amax reduction = store_block_m × num_epilogue_warps × 4B
  - barriers (dispatch + tmem_full/empty + combine)
  - tmem_ptr (4 B)

Per-stage (× num_stages):
  - A tile: LOAD_BLOCK_M × BLOCK_K × 1B (FP8)
  - B tile: LOAD_BLOCK_N × BLOCK_K × 0.5B (FP4 unpacked view)
  - SFA, SFB: sf_block_m/n × 4 B
  - 2 × 8 B barriers

num_stages = (232448 - smem_fixed) / smem_per_stage, 要求 >= 2
```

## 2. Hopper smem 上限

- H100 opt-in dynamic smem 上限 228 KB（可能随 CUDA 版本略微调整）。
- 和 SM100 相差不大（~2%）。

## 3. Hopper 版本 smem 账本的变化

| 条目 | SM100 | Hopper (方案 A, FP8 权重, 单 CTA) | 说明 |
|---|---|---|---|
| A tile per stage | `LOAD_BLOCK_M × 128` (1B) | `BLOCK_M × 128` (1B) | 无 2-CTA multicast 就没有 LOAD_BLOCK_M=BLOCK_M/2 |
| B tile per stage | `LOAD_BLOCK_N × 128 × 0.5` (FP4) | `BLOCK_N × 128` (1B, FP8) | **权重从 FP4 → FP8，每 stage B tile 2×**|
| SFA per stage | `sf_block_m × 4` | `BLOCK_M × 4` (no padding) | 不需要 128-align |
| SFB per stage | `sf_block_n × 4` | `BLOCK_N × 4` | 不需要 128-align，少一点 |
| 累加器存储 | TMEM (不占 smem) | **register (不占 smem)** | 相同不占 smem，但吃寄存器 |
| Output buffer | L1 FP8 × 2 stages | L1 FP8 × 1 stage（epilogue in-place，无跨 warp producer-consumer） | 可能减半 |
| barriers | tmem_full/empty + ... | wgmma 通过 commit/wait 不吃 barrier slot | barrier slot 数量少 |

**关键变化**：权重从 FP4 (0.5B) 变成 FP8 (1B)，每 stage 的 B tile 占用翻倍。

### 3.1 估算表（BLOCK_M=128, BLOCK_N=128, BLOCK_K=128）

**SM100 情形**（LOAD_BLOCK_M = 64 because of 2-CTA）：
- A: 64×128 = 8 KB, B: 128×128×0.5 = 8 KB, SFA+SFB ~0.6 KB, barriers 16 B
- Per-stage ≈ 16.6 KB
- smem_fixed ≈ 20 KB
- num_stages = (232 - 20)/16.6 ≈ 12

**Hopper 情形**（单 CTA, FP8 权重）：
- A: 128×128 = 16 KB, B: 128×128 = 16 KB, SFA+SFB ~0.6 KB, barriers 16 B
- Per-stage ≈ 32.6 KB
- smem_fixed ≈ 18 KB（barriers 少一些，无 tmem_ptr）
- num_stages = (228 - 18)/32.6 ≈ 6

### 3.2 估算表（BLOCK_M=64, BLOCK_N=128, BLOCK_K=128）

Hopper：
- A: 64×128 = 8 KB, B: 128×128 = 16 KB, SF ~0.5 KB
- Per-stage ≈ 24.5 KB
- num_stages = (228 - 18)/24.5 ≈ 8

### 3.3 启用 cluster + multicast（策略 6-A）

如果保留 cluster=2 和 multicast A，等效 per-CTA A 负担减半：
- A per CTA: 64×128 = 8 KB (仅储存一半，因为每个 CTA 只处理一半的 M)
- **等效**回到 SM100 量级
- num_stages 与 SM100 相近（~10-12）

所以 **cluster + multicast 在 Hopper 上仍然是关键的 stage 数恢复手段**。

## 4. Pipeline 深度对性能的影响

pipeline 的 stage 数直接决定了 TMA producer 能超前几步：
- **stage=2**：producer 只能比 consumer 超前 1 步，TMA / MMA 完全串行化，性能差。
- **stage=4-6**：可以掩盖 TMA load 的 latency（HBM ~400 ns, L2 hit ~150 ns），性能还行。
- **stage≥8**：有足够 slack 掩盖 overhead，compute-bound 和 memory-bound 的 overlap 充分。

SM100 Mega MoE 在 BLOCK_M=128 下能跑到 ~12 stages；Hopper 单 CTA 配置掉到 ~6 stages。这对 MMA-TMA overlap 是一个明显的 regression，特别是在 token-sparse 场景（L1 pool 等 arrival 占比高时，stage 多的意义更大）。

**恢复方式**：
1. 启用 cluster=2 + multicast（策略 6-A） → stage 恢复到 ~10。
2. 缩小 BLOCK_N（例如 64） → 每 stage 小，stage 多；但 WGMMA 吞吐降低，有 trade-off。
3. 方案 B（kernel 内 FP4 反量化）→ 权重 stage 占 0.5B，stage 数可以翻倍，但反量化开销 + 寄存器挤压。

## 5. 动态 smem 的组织

Hopper 的动态 smem 分配和 SM100 完全一样（`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)`）。改动在两个点：
1. `smem_fixed` 中**去掉 tmem_ptr（4 B）和 tmem_full/empty barriers**。
2. `smem_fixed` 中的 **L1 output buffer 减半**：SM100 用 2 stages（一个给当前 epilogue，一个给并行的 epilogue-next），Hopper 因为是单 warpgroup in-place epilogue，只需 1 stage。

## 6. Shared memory 和 L1 arrival / L2 arrival

Dispatch → MMA 之间的同步靠 `l1_arrival_count` 和 `l2_arrival_mask` 在 **global workspace** 里维护，不占 shared memory。这一部分不变。

## 7. Barrier slot 规划

SM100 的 mbarrier 分配：
```
barrier_start_ptr[0..3]               = dispatch barriers (kNumDispatchWarps=4)
barrier_start_ptr[4..4+2N]            = full/empty barriers × kNumStages
barrier_start_ptr[4+2N..4+2N+2×2]     = tmem_full/empty barriers × 2 (epilogue stages)
barrier_start_ptr[...]                = combine barriers × 2 × kNumEpilogueWarps
```

Hopper：
```
barrier_start_ptr[0..3]        = dispatch barriers
barrier_start_ptr[4..4+2N]     = full/empty barriers × kNumStages
barrier_start_ptr[...]         = combine barriers × 2 × kNumEpilogueWarps
```
删掉 tmem_full/empty。每个 barrier 8 B，节省 ~32-64 B，微量。

## 8. Pipeline 流水线角色映射

对齐 02 章的 warp 角色建议：

```
Hopper Mega MoE 线程布局（单 cluster，以 BLOCK_M=128 为例）:
┌──────────────────────────────────────────────────────────────┐
│ CTA 内 256 thread, 即 2 warpgroup                            │
│                                                              │
│ Warpgroup 0 (warp 0-3, 128 thread, 80 reg/thread):           │
│   Producer:                                                  │
│   - dispatch pull (在 MMA 开始前)                            │
│   - TMA load A/B/SFA/SFB with mbarrier                       │
│   - final combine reduce (MMA 完成后)                        │
│   - workspace 清理                                           │
│                                                              │
│ Warpgroup 1 (warp 4-7, 128 thread, 240 reg/thread):          │
│   Consumer:                                                  │
│   - WGMMA issue (整 warpgroup)                               │
│   - CUDA promotion (SF × accum)                              │
│   - SwiGLU + topk_weight + amax + FP8 cast                   │
│   - TMA store to l2_acts                                     │
│   - 写 l2_arrival_mask bit                                   │
│   - L2 GEMM (同样 WGMMA)                                     │
│   - L2 epilogue: BF16 cast + NVLink write-back               │
└──────────────────────────────────────────────────────────────┘
```

与 SM100 的 12-warp 布局对比（4 dispatch + 1 acts TMA + 1 weights TMA + 1 UMMA + 1 idle + 4-8 epilogue），Hopper 压缩到 **2 个 warpgroup**，角色合并、overlap 变差但代码结构清晰。

如果性能不够、寄存器够用，可以进一步升级到 **3 个 warpgroup**：
- WG0: producer（TMA + dispatch）
- WG1: consumer math (WGMMA + promotion)
- WG2: consumer epilogue (SwiGLU + quant + store)

WG1 和 WG2 通过 smem 传 accum / BF16，解耦 math 和 epilogue 的寄存器需求。这是 CUTLASS Hopper 的 "warp-specialized" epilogue 模式，性能最佳但实现复杂，作为 v2 优化目标。

## 9. 小结

- Hopper smem 上限 228 KB 与 SM100 基本持平。
- 权重从 FP4 → FP8 导致每 stage 的 B tile 翻倍，**stage 数从 ~12 掉到 ~6**。
- 恢复 stage 数的首要手段是 **启用 cluster=2 + multicast A**。
- 去掉 TMEM 相关 barrier/buffer 后 smem_fixed 略减。
- 推荐 Hopper v0 用 **2 warpgroup 结构**（producer + consumer），v2 升级到 **3 warpgroup（warp-specialized epilogue）**。
- BLOCK_M 上限 128，BLOCK_N 可选 64/128/256；候选集需要重新枚举。
