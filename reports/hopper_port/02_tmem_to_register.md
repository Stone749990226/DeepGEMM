# 02 · Tensor Memory → 寄存器累加器的重构

> **难点强度：★★★★★**（整个移植最核心的结构性改动）

## 1. SM100 上的 TMEM 用法

在 [sm100_fp8_fp4_mega_moe.cuh](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) 中，tensor memory 承担三类角色：

```
TMEM 列分配（kNumTmemCols = align(kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols)）
┌──────────────────────────────────────────────────────────────┐
│ [0 .. kNumAccumTmemCols)                                     │
│   累加器双缓冲：UMMA_N × kNumEpilogueStages=2               │
│   - 每个 stage 占 UMMA_N(=BLOCK_M) 列，每列装一个 FP32 accum │
│   - 由 UMMA warp 写，epilogue warps 读                       │
├──────────────────────────────────────────────────────────────┤
│ [... + kNumSFATmemCols)                                      │
│   SFA tensor memory 区：UTCCP 从 smem 写入的 scale factor    │
│   （4×32 dp128bit_2cta，被硬件直接消费）                     │
├──────────────────────────────────────────────────────────────┤
│ [... + kNumSFBTmemCols)                                      │
│   SFB tensor memory 区                                       │
└──────────────────────────────────────────────────────────────┘
```

关键调用点：
- [line 65](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L65)：`using Allocator = cute::TMEM::Allocator2Sm;`
- [line 309](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L309)：`Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);`
- [line 863](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L863)：UMMA 把结果写入 `accum_stage_idx * UMMA_N` 起始的 TMEM 列
- [line 981/1145](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L981)：`SM100_TMEM_LOAD_16dp256b1x` 每次从 TMEM 读 8 个 FP32 到 8 个 lane 的寄存器
- [line 1227](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1227)：kernel 末尾 `Allocator().deallocate(...)`

## 2. Hopper 上 WGMMA 的累加器位置

SM90 没有 TMEM。WGMMA 指令的累加器直接存储在**每个线程的寄存器**中：

```
WGMMA m64n_k32 FP8 BF16:
  - 一个 warpgroup (128 thread) 共同持有 64×N 的 FP32 累加器
  - 每个 thread 持有 (64×N)/128 = N/2 个 FP32 (对 BLOCK_N=128 是 64 个 FP32)
  - BLOCK_M=128 时 warpgroup 叠两次 → 每 thread 128 FP32 = 512 B 的 accumulator register
```

从 [sm90_fp8_gemm_1d1d.cuh](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh) 可见，SM90 FP8 GEMM 直接在栈上开 `float accum[WGMMA::kNumAccum]` 数组，所有累加存在这里。

## 3. 迁移会出现的 4 个问题

### 3.1 寄存器压力爆炸

SM100 的 TMEM 是「物理上独立」的累加器存储，SM 的 64K 寄存器文件几乎不需要拿出来给 accum。移到 SM90 后：

| 配置 | 累加器 register (per thread) | 累加器总量 |
|---|---|---|
| BLOCK_M=128, BLOCK_N=128, 双缓冲 | 2×(128×128)/128 = 256 FP32 | 1 KB / thread |
| BLOCK_M=192, BLOCK_N=128, 双缓冲 | 2×(192×128)/128 = 384 FP32 | 1.5 KB / thread |

在 SM100 上 epilogue warps 是 208 reg/thread；在 Hopper 上如果仍然想做双缓冲，**仅累加器就要吃 256-384 个寄存器**，一个 warpgroup 只剩 128-240 reg 给中间变量、SwiGLU 计算和 scale factor。H100 的一个 warpgroup 上限 256 reg/thread（`setmaxnreg`），双 stage 的 BLOCK_M=192 几乎不可行。

**对策**：
1. 放弃双 stage accumulator：Hopper 版本的 MMA 和 epilogue 共享同一批寄存器，WGMMA 完成后立刻 epilogue，不做 accumulator 双缓冲（等价 `kNumEpilogueStages = 1`）。
2. 在 BLOCK_M 候选集中剔除 192，最大 128，限制每 thread 累加器 ≤ 128 FP32。
3. 考虑让 MMA 和 epilogue 复用同一个 warpgroup（Hopper 常见做法：producer warp + consumer warpgroup），避免再切出专门的 UMMA warp（当前 SM100 是 warp 6 独占）。

### 3.2 MMA 与 epilogue 的生产-消费模型瓦解

SM100 当前模型：
```
UMMA warp (writer) ──tmem_full──> Epilogue warp (reader)
   ↑                                    │
   └─────── tmem_empty ─────────────────┘
```
这是**跨角色** producer/consumer：UMMA 发射后不等待，epilogue 在 TMEM 上读。

Hopper 上累加器在寄存器中，跨 warp 不能直接访问别人寄存器。有两种等价形态：

**形态 A：WGMMA warpgroup 自己做 epilogue**
```
[warpgroup N]  wgmma.mma_async(A, B, d)       // d 在本 warpgroup 寄存器
               wgmma.commit_group
               wgmma.wait_group<0>
               // epilogue in-place (SwiGLU, quantize, store)
```
简单但失去了 MMA 和 epilogue 的 overlap（MMA 打完才 epilogue，当前 SM100 是 overlap 的）。

**形态 B：双 warpgroup ping-pong**（CUTLASS sm90 gemm 常见）
```
warpgroup 0: MMA wave i   → epilogue wave i   → MMA wave i+2 ...
warpgroup 1:                MMA wave i+1      → epilogue wave i+1 ...
```
需要两个 warpgroup 轮流做 MMA+epilogue，smem 分两份 accumulator tile。但 Mega MoE 的 epilogue 已经比较复杂（SwiGLU + amax + 量化），再 ping-pong 会让 warp 角色 + register budget 非常拥挤。

**决策建议**：先走形态 A（simpler，先 correct 再 fast）；等稳定后再上 ping-pong。这意味着 SM100 当前的 `tmem_full_barriers / tmem_empty_barriers` 直接删掉，换成 WGMMA 的 `commit_group/wait_group`。

### 3.3 `SM100_TMEM_LOAD_16dp256b1x` 的 gate/up 排列特性消失

SM100 上 L1 epilogue 利用了一个非常微妙的特性：`SM100_TMEM_LOAD_16dp256b1x` 一次返回 8 个 FP32，排列成
```
values[0..7] = [g0, g1, u0, u1, g2, g3, u2, u3]
```
**gate/up 交错对齐**，刚好匹配 Python 端 `_interleave_l1_weights(gran=8)` 的权重布局。kernel [line 943](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L943)：
```cpp
// With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
// (values[0], values[2]), (values[1], values[3]), (values[4], values[6]), (values[5], values[7])
```

Hopper 上 WGMMA 的累加器分布是由 PTX 手册规定的 WGMMA C 矩阵 layout：每 thread 持有的 FP32 不是 gate/up 交错的，而是**列方向连续**。原本的 gate/up 配对（`values[0]` 配 `values[2]`）在 WGMMA layout 下**不再成立**。

**对策**：
1. **取消 L1 权重 interleave**：移植版本的 `transform_weights_for_mega_moe` 把 `_interleave_l1_weights` 改成 no-op，权重仍然是 `[gate | up]` 顺序。
2. **epilogue 里用 warp shuffle 做 gate/up 配对**：N 方向连续分布，gate 和 up 在 warpgroup 内不同 lane，需要 `shfl.sync` 把 up 的值取回 gate 所在 lane。代价是额外的 shuffle 带宽。
3. **另一种思路**：按 N 维切块，先算完所有 N 方向的 gate（写 smem），再算 up（写 smem），最后从 smem 合并。开销大，但逻辑清晰。

从 H100 实际性能角度看，因为 WGMMA 本来会在 warp 内提供 64×16 的 tile per instruction，可以让 `BLOCK_N = gate_N + up_N`，比如 `gate_N = 128, up_N = 128, BLOCK_N = 256`，对 K 维做一次 WGMMA 得到 gate 和 up 各自的累加器子集，epilogue 里成对处理 —— 这里没有 SM100 那种「硬件天然交错」的便利，但工程上可以接受。

### 3.4 Accumulator 不能跨 block_phase 复用

SM100 的 TMEM accumulator 是 per-CTA 的，L1 和 L2 block 都用同样的 TMEM 区域，只是时间上切换；而 Hopper 寄存器是 per-thread，L1 和 L2 想用不同 BLOCK_N 时，寄存器数量会变：

- L1: `BLOCK_N = 128`（gate）+ 128（up）= 256？或者 `BLOCK_N = 128`，gate/up 共享？
- L2: `BLOCK_N = 128`

需要让 epilogue 之间 **寄存器 alias**（例如用 union 或两个独立的 scope）避免同时占用。这在 CUDA 里实践上通常通过「epilogue 结束后大括号出作用域」让 nvcc 回收寄存器，但不是 100% 可靠。

## 4. 推荐的新累加器组织

```
Hopper Mega MoE L1 GEMM:

Warpgroup 0 (128 thread, 256 reg/thread):
  for each block assigned by scheduler:
    1. wgmma.mma_async for K blocks (accum in float[kNumAccum])
    2. wgmma.wait_group<0>
    3. apply SFA × SFB CUDA promotion (chapter 05)
    4. SwiGLU + amax + topk_weight
    5. FP8 cast
    6. st.shared (packed U8x4)
    7. bar.sync warpgroup
    8. TMA store to l2_acts + SF write
    9. set l2_arrival_mask bit

Warpgroup 1 (TMA / dispatch / combine, 128 thread, 48-80 reg/thread):
  - TMA A/B/SF load producer
  - Dispatch pull (可与 MMA warpgroup 并行)
  - Final combine reduce
```

换句话说，**Hopper 版本的 warp 角色会从 SM100 的 4+3+1+2×4 = 12 warp 架构，压缩到「producer warpgroup + consumer warpgroup」的 2-warpgroup 经典 Hopper 模式**，这是第二大结构性改动。

## 5. 小结

- TMEM 的消失让累加器被迫移进寄存器，直接导致 BLOCK_M ≤ 128 且取消双 stage。
- UMMA/Epilogue 的跨角色 producer/consumer 被迫退化为单 warpgroup in-place epilogue，损失一些 overlap。
- `SM100_TMEM_LOAD_16dp256b1x` 带来的 gate/up 硬件交错特性失效，Python 权重 transform 需要改，epilogue 里要引入 shuffle 或 N-块拆分。
- 推荐采用 Hopper 经典的 producer/consumer warpgroup 结构，而不是照搬 SM100 的 12 warp 架构。
