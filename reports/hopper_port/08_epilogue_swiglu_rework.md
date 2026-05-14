# 08 · L1 Epilogue：SwiGLU / amax / FP8 量化 的 Hopper 重构

> **难点强度：★★★★☆**

L1 epilogue 是整个 kernel 里精度与性能最敏感的部分。SM100 版本依赖 TMEM 的 `SM100_TMEM_LOAD_16dp256b1x` 指令的 gate/up 交错特性、跨角色 TMEM producer/consumer 模型、以及 2-CTA 结构，**每个都需要在 Hopper 上找替代**。

## 1. SM100 L1 Epilogue 逻辑链回顾

按 [mega_moe_code_analysis.md §10](reports/mega_moe_code_analysis.md) 和 [sm100_fp8_fp4_mega_moe.cuh:927-1000](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)：

```
per ATOM_M = 8 行:
  1. TMEM_LOAD_16dp256b1x → 8 个 FP32 values (gate/up 交错: (v0,v2),(v1,v3),(v4,v6),(v5,v7))
  2. 跨 32 token 从 l1_topk_weights_buffer 读一次 topk_weight，warp shuffle 分发
  3. 对每对 (gate,up):
       gate_f32 = bf16_to_float(values[i])
       up_f32   = bf16_to_float(values[i+2])
       # 可选 clamp
       silu_gate = gate_f32 / (1 + exp(-gate_f32))
       swiglu    = silu_gate * up_f32 * topk_weight
  4. amax reduction 4-lane group → smem_amax_reduction
  5. 跨 warp 合并 amax（两个 warp 共享同一组 M 行）
  6. 计算 FP8 E4M3 的 (sf, sf_inv)
  7. cast_to_fp8_e4m3(swiglu * sf_inv)
  8. STSM (SM100_U8x4_STSM_T) 写 smem
  9. warpgroup sync
 10. TMA store 到 l2_acts, 宽度 BLOCK_N/2
 11. 写 SF 到 l2_sf_buffer (4×32 UTCCP 布局)
 12. tma_store_wait
 13. red.or.rel.gpu l2_arrival_mask |= (1<<n_block_idx)
```

## 2. Hopper 重构的四个关键差异

### 2.1 Accumulator 的取法 —— 没有 TMEM_LOAD

WGMMA 的累加器已经在寄存器里，所以步骤 1 直接变成 `float* accum = final_accum`（05 章的 CUDA-promotion 结果）。

但 **WGMMA 的 accumulator 分布** 和 SM100 TMEM_LOAD 很不一样。PTX 手册 Table 33 描述 WGMMA `m64nNk32 FP32` 输出的 register fragment：
- 一个 warp 的 32 个 lane 每 lane 持有 `N/4` 个 FP32。
- 每 8 行（M 方向）在 warp 内由 lane 0..3 的第 0/1 行、lane 4..7 的第 2/3 行……分布。
- **列方向（N）连续**：lane 0 的 FP32[0..N/4-1] 是同一行（取决于 warp 的 M 偏移）里 N 维连续的值。

对 SwiGLU 而言，gate 和 up 是**在 N 维度上分离**的（前半 gate，后半 up）。在 SM100 的 TMEM_LOAD layout 下，一条 load 指令刚好能拿到 `(g0..g3, u0..u3)` 的 8 个 FP32（交错），所以 gate/up 配对是硬件自带的。Hopper 上，`accum[0..N/4]` 是**整段连续 N 中 lane 负责的 N 切片**，gate 和 up 要么在同一 lane 的不同 index，要么在不同 lane 的相同 index，取决于 N 的切分方式。

### 2.2 Gate/Up 配对策略

三种可能的组织：

**策略 1：N 切成 gate 半 + up 半，分别 WGMMA**
```
BLOCK_N_gate = N/2   (gate weight slice)
BLOCK_N_up   = N/2   (up   weight slice)
每次 WGMMA 输入是完整 N（等效 SM100 的 BLOCK_N），
但用两个 accumulator array: accum_gate[N/8], accum_up[N/8]（每 warp lane）

epilogue:
  for i in range(N/8):
    gate = accum_gate[i]
    up   = accum_up[i]
    swiglu = silu(gate) * up * topk_weight
    ...
```
**优点**：gate/up 在同一 lane 配对，无需 shuffle。
**缺点**：需要发两次 WGMMA（或在 K 维同时做 gate 和 up，N 维分两次 issue），accumulator 数量 × 2，寄存器压力加大。

**策略 2：N 连续，gate/up 在 N 维方向拆分**
```
BLOCK_N = N
gate 占 accum[0..N/8], up 占 accum[N/8..N/4]
epilogue:
  for i in range(N/8):
    gate = accum[i]
    up   = accum[N/8 + i]
    swiglu = silu(gate) * up * topk_weight
    ...
```
**优点**：一次 WGMMA issue，accumulator 数不变。
**缺点**：gate 和 up 是连续 N 的两半，在寄存器中**同属一个 lane**，但 index 范围不同。遍历时 lane 要同时读两处 index，寄存器拷贝可能不合并，但整体合理。**这是推荐做法**。
**关键**：权重张量必须按 `[gate_N | up_N]` 顺序排列（就是原始 MoE 权重的默认顺序），**不做 gate/up interleave**（与 SM100 不同）。

**策略 3：用 warp shuffle 交换 gate/up**
Gate 在 lane A，up 在 lane B（两 lane 由交错权重决定），用 `shfl.sync` 把 up 值广播到 lane A 后再组对。
**缺点**：带宽额外开销、实现复杂。

**决策**：采用 **策略 2**。Python 端 `transform_weights_for_mega_moe_sm90` 保持 `[gate | up]` 原始顺序，kernel 中 epilogue 按 `i vs N/8+i` 配对。

### 2.3 Amax reduction 的组织

SM100 的 amax reduction 在 smem 上做 4-lane group。Hopper 同样可以做，但更常见的做法是**先在 warp 内 `__reduce_max_sync`，再在 warpgroup 内 smem 合并**。

Hopper amax reduction 伪代码：
```cpp
// step 1: 本 lane 的 N/8 个 swiglu 值 local amax
float amax = 0.f;
for i in range(N/8):
    amax = max(amax, fabsf(swiglu[i]));

// step 2: 同一行 M 的不同 lane 间 warp reduce
// WGMMA 分布：同一行对应的 lanes 是 lane 0..3 (第 1 行), lane 4..7 (第 2 行), ...
// 每 4 个 lane 共享一行的 amax
#pragma unroll
for (offset = 1; offset < 4; offset *= 2)
    amax = max(amax, __shfl_xor_sync(0xffffffff, amax, offset));

// step 3: 根据需要跨 warp（两个 warpgroup 合并 M 区间时）通过 smem
smem_amax[row_idx] = amax;
__syncthreads();
amax = smem_amax[row_idx];  // 再读一次
```

关键要对齐 **WGMMA C 矩阵的 lane→row 映射**，否则 amax 会按错误的行分组。参考 CUTLASS `gmma_mma_accum_smem.hpp` 里的 layout。

### 2.4 FP8 cast 指令

SM100 用 `SM100_U8x4_STSM_T`（unsigned 8-bit × 4, store shared memory, transposed）。
Hopper 没有 `STSM_T` 的 FP8 对应变体，但有：
- `stmatrix.sync.aligned.m8n8.shared::cta.b16`（BF16 STSM）可用
- `st.shared.v4.u32` 简单 4-dword 写

对 FP8 输出，最实用的是**手动 packed st.shared**：
```cpp
uint32_t packed0 = pack_4_fp8(swiglu[0..3]);  // 4 fp8 → 32-bit
uint32_t packed1 = pack_4_fp8(swiglu[4..7]);
asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(addr), "r"(packed0), "r"(packed1));
```

STSM 可以做 transpose；如果 shared memory 的目标 layout 需要 M/N 转置，要么用 `stmatrix.trans`（BF16 版本有），要么先写 smem 再 warp shuffle 重排。

**复杂度警告**：SwiGLU 完的 FP8 TMA store 到 l2_acts 需要 swizzled layout（TMA 64B swizzle），所以 epilogue 写 smem 必须按 TMA swizzle 约定，**不是随便的 linear layout**。这一步是 Hopper FP8 kernel 的通用难点，好在 DeepGEMM 的 `sm90_store_cd.cuh` 类似思路已经解决过（虽然不是给 SwiGLU 的）。建议参考 CUTLASS 的 `EpilogueHopperTmaWarpSpecialized` 实现。

## 3. Topk weight 的流动

SM100：
```cpp
// 每 32 token 从 l1_topk_weights_buffer 加载一次 weight 到寄存器
// 用 ptx::exchange 广播到需要 lane
```

Hopper 保持完全相同的逻辑。`l1_topk_weights` 是 dispatch 阶段拉过来的 `[num_pool_tokens, 1]` float 数组，L1 epilogue 中按 ATOM_M 行一次读 ATOM_M 个 weight，通过 `__shfl_sync` 分发。这一块没有硬件依赖，直接移植。

## 4. L2 epilogue：BF16 cast + 远端写回

L2 epilogue 逻辑相对简单（[§11](reports/mega_moe_code_analysis.md)）：
```
TMEM_LOAD → cast to BF16 → STSM → NVLink store 到远端 combine_buffer
```

Hopper 对应：
```
final_accum (FP32 in register) → cast to BF16 → st.shared (packed) → NVLink store
```

差异：
- 去掉 TMEM_LOAD，直接用 `final_accum`。
- STSM 用 BF16 版本 `stmatrix.sync`，Hopper 原生支持。
- NVLink store 写远端 combine_buffer（通过 symmetric memory map）— 与 SM100 一样，不动。

L2 epilogue 的相对简单主要是因为**不做量化**。

## 5. 寄存器占用重新测算

Hopper L1 epilogue 一次处理 `BLOCK_M × BLOCK_N` 的 tile。以 BLOCK_M=64, BLOCK_N=256（gate+up 合并后逻辑 N=128）为例：
- 每 warp（32 lane）持有 `64 × 128 / 32 = 256 FP32` → 每 lane 8 FP32 accum，再加 `final_accum` 同样 8 FP32 → 16 FP32 = 64 B / thread。
- 如果 BLOCK_M=128，翻倍到 32 FP32 = 128 B / thread。

加上 SwiGLU 中间变量（gate/up/silu/swiglu = 4×8 FP32 + BF16 转换）、amax reg、SF load reg、epilogue 地址/索引 reg，按经验一个 Hopper FP8 GEMM + SwiGLU 的 consumer warpgroup 大约 **200-240 reg/thread**。这在 `setmaxnreg=240` 下勉强够，但几乎没有余量。

**如果 BLOCK_M 取 192**：accumulator 翻 1.5×，寄存器明显超标。所以 08 章 + 02 章的结论一致：**Hopper BLOCK_M 上限 128**。

## 6. 小结

- L1 epilogue 的**每一步硬件依赖**都要替换：TMEM_LOAD → register、UTCCP SF → ld.shared、U8x4_STSM_T → 手写 packed store、2-CTA epilogue → 单 CTA epilogue。
- Gate/up 配对策略推荐 **N 维连续两半切分（策略 2）**，Python 端去掉 `_interleave_l1_weights`。
- Amax reduction 用 WGMMA C 的 lane→row 分布做 warp reduce；跨 warpgroup 用 smem。
- FP8 store 到 swizzled smem 是 Hopper 通用难点，参考 CUTLASS Epilogue 实现。
- L2 epilogue 相对简单（不量化），主要是 BF16 cast + NVLink 写回，移植成本低。
- 寄存器预算重新计算后，BLOCK_M 上限 ≤ 128。
