# 03 · UMMA → WGMMA：MMA 指令语义差异

> **难点强度：★★★★☆**

本章聚焦把 SM100 UMMA 指令（`SM100_MMA_MXF8F6F4_2x1SM_SS`）替换为 SM90 WGMMA（`wgmma.mma_async`）时，指令级的**语义差异**及其对代码组织的连锁影响。

## 1. 指令语义对照

| 维度 | SM100 UMMA | SM90 WGMMA |
|---|---|---|
| 指令族 | `SM100_MMA_MXF8F6F4_2x1SM_SS` / `1x1SM` | `wgmma.mma_async.sync.aligned.m64nNk32` |
| 发射单位 | **single thread**（由 leader CTA 的一个 warp 代表整个 cluster） | 整个 warpgroup (128 thread) 集体发射 |
| Issue 异步性 | 异步，`umma_arrive` 到 mbarrier | 异步，`wgmma.commit_group` + `wait_group` |
| A 操作数来源 | Shared memory（SS 版本）或 TMEM（TS 版本）| **寄存器或 shared memory**（RS / SS） |
| B 操作数来源 | Shared memory | Shared memory |
| 累加器位置 | **TMEM** | **寄存器**（warpgroup 内分布）|
| 单指令 M/N/K 尺寸 | `MXF8F6F4` 2x1SM: M∈{64,128,256}, N∈{8..256}, K=32 | `m64nNk32` fixed: M=64, N∈{8..256 step 8}, K=32 |
| 精度组合 | 混合 FP8/FP6/FP4 | FP8 × FP8 / FP16 × FP16 / BF16 / TF32 |
| Block scaling | ✅ 硬件消费 SFA/SFB（UE8M0） | ❌ 没有 block-scaled 变体 |
| 2-CTA 合作 | ✅ `2x1SM`：两个 CTA 的 A tile 自动共享 | ❌ 单 warpgroup，cluster 仅靠 TMA multicast |
| 完成通知 | `umma_arrive` → mbarrier | `wgmma.commit_group` → `wait_group<n>` |
| scale_d 语义 | UMMA 描述符里 `ScaleOut::Zero/One` | WGMMA 指令参数 `ScaleOut::Zero/One`（每次 issue 决定）|

## 2. Mega MoE 中的具体改动点

### 2.1 Issue 方式

当前代码（SM100）：
```cpp
// kernel line 854-867（简化）
for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
    auto a_desc = ...;  // K-offset 更新
    auto b_desc = ...;
    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
        b_desc, a_desc, accum_stage_idx * UMMA_N,
        instr_desc, /*scale_d=*/k == 0 ? 0 : 1);
}
```

重点：
- 只有 leader CTA 的一个 warp 发射（`is_leader_cta` 分支里），实际只有 1 个 thread 写指令。
- 累加器地址是一个 TMEM 列号（`accum_stage_idx * UMMA_N`）。
- `scale_d=0` 第一次清零、`=1` 后续累加。

Hopper WGMMA 改成：
```cpp
// Hopper pseudo-code
float accum[kNumAccum];  // 每 thread 的累加器
warpgroup_fence_operand(accum);
#pragma unroll
for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++k) {
    WGMMA::wgmma(desc_a_k, desc_b_k, accum, /*scale_d=*/k != 0);
}
warpgroup_commit_batch();
warpgroup_wait<0>();
warpgroup_fence_operand(accum);
```

语义区别：
- **整个 warpgroup 必须同时发射**，不能单线程代劳；MMA warp 就不再是一个独立 warp。
- 累加器不再是 TMEM 地址，而是每个 thread 的寄存器数组 `accum`。
- `warpgroup_fence_operand` 是 WGMMA 特有的 fence，需要在 accumulator 被其他指令读/写前后加。

### 2.2 描述符构造

SM100 当前：
```cpp
auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
    a_dtype, b_dtype, sf_dtype, float,
    UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
auto a_desc = mma::sm100::make_umma_desc<...>(smem_a, 0, 0);
auto b_desc = mma::sm100::make_umma_desc<...>(smem_b, 0, 0);
```

Hopper 等价（已存在于仓库 [sm90.cuh](deep_gemm/include/deep_gemm/mma/sm90.cuh)）：
```cpp
using WGMMA = mma::sm90::FP8MMASelector<BLOCK_N>::type;
auto desc_a = mma::sm90::make_smem_desc<K_MAJOR, ...>(smem_a, ...);
auto desc_b = mma::sm90::make_smem_desc<K_MAJOR, ...>(smem_b, ...);
```

不同点：
- **没有 block-scaled 描述符变体**。SF 走软件路径（见 05）。
- WGMMA 的 swizzle mode 由描述符里的 `LayoutType` 指定（B32/B64/B128），而 UMMA 是通过 `kSwizzleAMode` 模板参数传入。语义一致但 API 不同。
- **没有 sparse MMA 和 FP4**，所以 `a_dtype, b_dtype` 只能是 FP8 E4M3 / E5M2。

### 2.3 Swap A/B 的含义改变

SM100 Mega MoE 使用了 **A/B swap**：权重当 A、激活当 B，目的是让 `UMMA_M = LAYOUT_AD_M * 2 = 256`、`UMMA_N = BLOCK_M`。这样 M 维度上可以做 2-CTA 合作（见 06）。

Hopper 没有 2-CTA MMA，A/B swap 的必要性消失。但对 grouped MoE 而言，swap 仍有价值：**A tile (weights) 相对稳定（同一个 expert 共享）、B tile (acts) 每次不同**。swap 后权重走 A、acts 走 B，可以让 B 的 TMA 更频繁、A 的 TMA 可复用。

建议 Hopper 版本**保留 swap**，但 MMA 语义变成：
```
A = weights tile [BLOCK_N_wgmma, BLOCK_K]      (FP8)
B = acts tile    [BLOCK_M_wgmma=64, BLOCK_K]   (FP8)
C = [BLOCK_N_wgmma, BLOCK_M_wgmma]             (FP32 in registers)
```
这里 `BLOCK_N_wgmma` 是 WGMMA 指令的 N 维（原本 UMMA 的 `UMMA_N = BLOCK_M = 64~128`）。

### 2.4 `instr_desc` 的 N 动态更新

SM100 kernel 有这样一段：
```cpp
// line 806: Dynamic update of UMMA N based on effective M
update_instr_desc_with_umma_n(instr_desc, effective_m);
```
目的：M 方向 padding 的行数用 `UMMA_N` 的缩短表达，让 UMMA 不做无效乘法。

Hopper WGMMA 的 N 是固定 template 参数，**不能动态改**，要么 issue 一个不同 N 的 wgmma 实例、要么接受 padding 计算。后者简单：Mega MoE 的 padding 最多每 expert `BLOCK_M-1` 行，按 BLOCK_M=64 计算是小量浪费；前者需要 host 侧多打一套 kernel 实例，不现实。

**对策**：Hopper 版本接受 padding 乘法，保持每个 block 的 M 都是 `BLOCK_M`。有效 M 的边界由 epilogue 里判断 `row < valid_m` 来跳过 store。

### 2.5 `wgmma.wait_group` 的 N 参数

Hopper 支持多批未完成 MMA 排队：
```cpp
wgmma.commit_group;   // 封装一批 issue
wgmma.wait_group<N>;  // 等到 pending group ≤ N
```
这允许**连续发射多个 block 的 MMA 再统一 wait**。但当 N > 1 时，每个 pending batch 的 accum 都得占寄存器，寄存器预算再次吃紧。

**建议**：Hopper 版本 `wait_group<0>`，即一批 issue 完立刻等，accum 只有一份。这是 correctness > perf 的稳妥选择。

## 3. 保留 / 抛弃清单

| SM100 设施 | 是否保留 | 替代 |
|---|---|---|
| `SM100_MMA_MXF8F6F4_2x1SM_SS::fma` | ❌ | `cute::SM90::GMMA::MMA_64x{N}x32_F32E4M3E4M3_SS::fma` |
| `cute::UMMA::make_instr_desc_block_scaled` | ❌ | 不需要（没有 block-scaled）|
| `cute::UMMA::make_umma_desc` | ❌ | `cute::SM90::GMMA::smem_desc`（仓库已有 [sm90.cuh:108](deep_gemm/include/deep_gemm/mma/sm90.cuh#L108)）|
| `umma_arrive` | ❌ | `warpgroup_commit_batch` + `warpgroup_wait` |
| `is_leader_cta` 的 MMA 分支 | ❌ | 整 warpgroup 发射 |
| `update_instr_desc_with_umma_n` | ❌ | padding 接受 |
| Swap A/B 的思路 | ✅ | 沿用，但 MMA tile 大小不同 |
| K=32 的迭代 | ✅ | 沿用，WGMMA 也是 k=32 |
| BLOCK_K=128 | ✅ | 沿用 |

## 4. 对 MMA warp 角色的影响

SM100 的 warp 分布（引自 01 章总览）：
```
warp 0-3 dispatch    | warp 4 acts TMA | warp 5 weights TMA | warp 6 UMMA | warp 7 idle | warp 8-15 epilogue
```
Hopper 版本：
```
warpgroup 0 (warp 0-3)  producer: 全部 TMA load + dispatch pull
warpgroup 1 (warp 4-7)  consumer: WGMMA + Epilogue（同一 warpgroup）
（可选 warpgroup 2-3 作 epilogue/combine ping-pong）
```
这是因为 WGMMA 本身是 warpgroup 级别，没法像 UMMA 那样让单个 warp 代劳。具体 warp 布局在 09 章再详细讨论。

## 5. 小结

- UMMA → WGMMA 不是「换一行指令」，而是改变**发射粒度**（thread → warpgroup）、**累加器位置**（TMEM → register）、**完成通知**（mbarrier → wgmma group）、**N 动态性**（运行时 → 编译期）。
- Swap A/B 的思路沿用，tile size 要重标定。
- SM100 那种「MMA warp 独立发射，epilogue warp 独立消费」的异步 pipeline 必须塌回「warpgroup 内 MMA + epilogue 串行」。
- Hopper 的 `wgmma.wait_group<N>` 虽然支持多批 pending，但在寄存器压力下建议 `<0>`。
