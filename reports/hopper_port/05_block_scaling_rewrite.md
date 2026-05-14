# 05 · Block-scaled MMA 缺失 → CUDA promotion pipeline

> **难点强度：★★★★☆**

## 1. SM100 上硬件做了什么

SM100 UMMA 指令 `SM100_MMA_MXF8F6F4_2x1SM_SS` 是**block-scaled MMA**：硬件在计算每个 MMA tile 的时候，会**同时读取 SFA/SFB**（scale factor for A/B，以 UE8M0 或 UE4M3 存储），在 tensor core 内部完成
```
D = sum_k ( SFA[m, k_block] × A[m, k] × SFB[n, k_block] × B[n, k] )
```
SF 每 32 个 K 元素共享一个 scale（UE8M0 = 2^e 的指数）。SF 的物理位置是 **tensor memory**，由 `SM100_UTCCP_4x32dp128bit_2cta` 指令从 shared memory 拷入。

相关关键行：
- [line 781](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L781)：`make_instr_desc_block_scaled<...>()` 指定 SF 精度
- [line 840](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L840)：`SM100_UTCCP_4x32dp128bit_2cta` 把 smem SFA/SFB 拷到 TMEM
- [line 863](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L863)：`SM100_MMA_MXF8F6F4_2x1SM_SS::fma` 消费 TMEM 里的 SF
- kernel 模板参数 `rm=1, rn=1, rk=32`：recipe 决定了 SF 的 k granularity

SF 流水线是**纯硬件路径**，整个 MMA 中间没有 CUDA 端的 FP32 乘法介入。

## 2. Hopper 的做法：CUDA-side promotion

DeepGEMM 的 [sm90_fp8_gemm_1d1d.cuh](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh#L246-L308) 里的实现方式是经典的 **per-128-channel FP8 GEMM with CUDA promotion**：

```cpp
// 每个 K block 做完 WGMMA 后，用 SFA/SFB 做一次 FP32 乘加 promotion
for (k_block = 0; k_block < num_k_blocks; ++k_block) {
    wgmma(A[k_block], B[k_block], accum /* 每 K block 清零 */, false);
    wgmma.wait_group<0>();

    // 从 shared memory 读 SF（per 128 channel, per row）
    float scale_a_0 = ld_shared(smem_sfa + r_0);   // row r_0 的 SF
    float scale_a_1 = ld_shared(smem_sfa + r_1);
    float2 scale_b   = ld_shared<float2>(smem_sfb + ...);  // 两列的 SF

    // FP32 accumulator promotion (累加到 final_accum)
    final_accum[i*4 + 0] += scale_a_0 * scale_b.x * accum[i*4 + 0];
    final_accum[i*4 + 1] += scale_a_0 * scale_b.y * accum[i*4 + 1];
    final_accum[i*4 + 2] += scale_a_1 * scale_b.x * accum[i*4 + 2];
    final_accum[i*4 + 3] += scale_a_1 * scale_b.y * accum[i*4 + 3];
}
```

关键：
- **两套累加器**：`accum` 是每个 K block 内的临时累加（WGMMA 写入），`final_accum` 是跨 K block 的最终累加（加了 scale）。
- 每 128 个 K 元素做一次 promotion（DeepGEMM 称为 **1d1d scaling**：每个 SF 对应 M 维 1 行 × K 维 128 列）。
- SF 存在 shared memory，通过 `ld.shared.f32` 一次一次取。

## 3. Mega MoE 的 SF 布局差异

Mega MoE SM100 的 recipe 是 **`(rm=1, rn=1, rk=32)`**：SF 的 K 粒度是 32，而不是 128。原因是 FP4 的 dynamic range 小，32-channel SF 能控制精度。

移植到 Hopper 后，如果按方案 A 把权重转成 FP8：
- FP8 的 dynamic range 比 FP4 大，**128-channel SF 足够**。
- SF 粒度从 32 改到 128 后，SF 张量比原本小 4×，带宽和存储都更舒适。
- 激活 SF 同样用 per-128-channel（与仓库 `sm90_fp8_gemm_1d1d` 对齐）。

这也意味着：
- Dispatch 阶段拉过来的 `x_sf` 需要重新生成（不能直接复用 SM100 的 32-channel UE8M0 SF）。
- 或者在 symmetric buffer 里保持 32-channel SF，kernel 里按每 4 个 32-channel SF 做一次累加，仍然每 128 channel 一次 promotion。两种选择都可以，后者更通用，前者更快。

## 4. Hopper SF pipeline 重构

基于方案 A（FP8 weights, per-128-channel float SF），整个 SF 流水线如下：

```
Global Memory                        Shared Memory                         Register/Accumulator
─────────────                        ─────────────                         ───────────────────
sfa [experts, M, K/128] ──TMA load──> smem_sfa[stage][BLOCK_M]
                                       │
                                       └── ld.shared.f32 per lane ──> float scale_a_*
sfb [experts, N, K/128] ──TMA load──> smem_sfb[stage][BLOCK_N]
                                       │
                                       └── ld.shared.f32 per lane ──> float2 scales_b_*

                                       WGMMA (FP8 × FP8 → FP32 accum)
                                                                          │
                                                                          ▼
                                                                       float accum[N_per_thread]
                                                                          │
                                                                          └── final_accum += sa * sb * accum
```

L1 和 L2 各做一次这样的 pipeline，**互相独立**。

## 5. 和 SwiGLU epilogue 的交互

SM100 L1 epilogue 是从 TMEM load 累加器 → SwiGLU。在 Hopper 上，accumulator 直接就是 `final_accum`（promotion 后），所以 L1 epilogue 的第一步变成：

```cpp
// Hopper Mega MoE L1 epilogue 草图
for each N iteration:
    wgmma(...);
    wait;
    // CUDA promotion
    final_accum[...] += sa * sb * accum_wgmma[...];

// 所有 K block 完成后，final_accum 即 SwiGLU 输入
// （gate/up 的配对由 N 维切分决定，见 02 和 08 章）
apply_swiglu(final_accum, topk_weight);
quantize_to_fp8(final_accum);
tma_store(...);
```

相比 SM100 的 `TMEM_LOAD → SwiGLU → FP8 cast`，Hopper 的 promotion 已经**把 scale 应用好了**，epilogue 里直接处理 `final_accum`。

## 6. SF 精度：UE8M0 vs float

SM100 用 UE8M0（8-bit 无符号指数）存 SF，消耗 1 byte/scale，硬件加速消费。
DeepGEMM SM90 路径用 **float SF**（4 byte/scale），软件消费。权衡：

| 项 | UE8M0 | float (SM90 常用) |
|---|---|---|
| 存储 | 1 B | 4 B |
| SF TMA 带宽 | 1× | 4× |
| 数值精度 | 只能表示 2^e | 全精度 FP32 |
| CUDA promotion 代价 | 需先 `exp2f(sf)` 转 FP32 | 直接乘 |
| 与 Mega MoE 原 SF 一致性 | 继续用 UE8M0 | 需要重量化 |

**建议**：保持 DeepGEMM FP8 GEMM 的约定，使用 float SF，per-128-channel。这样一切软件栈可以复用。对 SF 大小敏感的场景可以切回 UE8M0，但 decode 代价（`exp2f` 或 bit hack）要算入 kernel。

## 7. 影响面总结

把 block-scaled MMA 拆成 CUDA promotion，带来的连锁改动：

1. **SF TMA 不再送到 TMEM**，目的地是 shared memory。UTCCP 不存在（07 章）。
2. SF 不再需要 `_transpose_sf_for_utccp` 变换（这个 Python 函数删除）。
3. SF 的物理 layout 重新设计，靠近 DeepGEMM sm90 的 per-128 channel 格式（07 章）。
4. MMA 循环里每个 K block 末尾都插入一段 FP32 promotion 代码，寄存器多占一份 `final_accum`。
5. L1 epilogue 的开头从「读 TMEM」改成「读 `final_accum`」，gate/up 配对逻辑重做（08 章）。
6. Scheduler 里的 SF 相关模板参数 (`kNumSFATmemCols`、`kNumUTCCPAlignedElems`、`SF_BLOCK_M` 等) 全部删除或改名。
7. Dispatch pull 阶段写的 `l1_acts_sf`、`l2_acts_sf` 布局改变（07 章）。

## 8. 小结

- SM100 的 block-scaled MMA 把 SF 消费塞进 tensor core，Hopper 不行。
- 替代方案是 DeepGEMM 已经验证过的 **per-K-block CUDA promotion**：`final_accum += SFA × SFB × accum`。
- 需要一份额外的 FP32 寄存器（`final_accum`），加剧寄存器压力。
- SF 粒度从 32 改到 128 可以和 DeepGEMM FP8 规范对齐，减少 SF 带宽压力。
- SF 在 Hopper 的物理路径：`GMEM → TMA → smem → ld.shared → register`，完全没有 TMEM/UTCCP。
