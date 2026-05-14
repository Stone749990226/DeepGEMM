# 07 · UTCCP 缺失后的 SF 布局重新设计

> **难点强度：★★★☆☆**

本章聚焦在 Scale Factor 的**物理布局**层面。硬件路径的改动（TMEM/UTCCP 去掉）在 02/05 已经讨论，这里专门说**如何重新组织 SF 数据结构**，让 Hopper CUDA promotion 能高效消费。

## 1. SM100 SF 布局回顾

SM100 Mega MoE 的 SF 布局有几层 quirk，源于 UTCCP 指令约束：

### 1.1 L1/L2 acts SF：`MN-major, 4×32 转置`

```
l1_acts_sf: int32 [num_padded_sf_pool_tokens, hidden / 128]
                    stride = {1, num_padded_sf_pool_tokens}   # MN-major
```

其中 `num_padded_sf_pool_tokens = (num_max_pool_tokens / block_m) * align(block_m, 128)`，即每个 BLOCK_M 大小的 block 在 M 维被填充到 128 的整数倍，满足 UTCCP 的 `4x32dp128bit` 拷贝模式。

kernel 侧 [line 127-132](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) 定义了 `transform_sf_token_idx`，把 `token_idx_in_expert` 映射到 padded SF 布局：

```cpp
uint32_t idx = token_idx_in_expert % BLOCK_M;
return token_idx_in_expert / BLOCK_M * SF_BLOCK_M
     + (idx & ~127u)           // 128 对齐基址
     + (idx & 31u) * 4         // 行 × 4
     + ((idx >> 5) & 3u);      // 列
```

这是 UTCCP 的 4×32 转置模式的反向映射。

### 1.2 权重 SF：`MN-major, 内部做 _transpose_sf_for_utccp`

Python 侧 `_transpose_sf_for_utccp` 对每 128 个 MN 元素做 4×32 转置，让 UTCCP 能把整段 SF 块按 warp lane 分派到 TMEM。

### 1.3 SF dtype：UE8M0（packed 为 int32）

4 个 UE8M0 打包成一个 int32，Mega MoE 用 `_PackedUE8M0` 形态在 GPU 和 TMA 上搬运。

## 2. UTCCP 消失后，SF 的约束完全放宽

Hopper 上 SF 由 `ld.shared.f32` 一个一个读，**不再需要 UTCCP 的 4×32 对齐**。这意味着：

- 不需要 `transform_sf_token_idx` 的 4×32 转置映射 → 删除。
- 不需要 padding 到 128 的 M 维对齐 → `num_padded_sf_pool_tokens` 简化为 `num_max_pool_tokens`。
- 不需要 Python 端 `_transpose_sf_for_utccp` → 改成 identity or 直接删除。
- SF dtype 可以从 UE8M0 改成 float（见 05 章讨论）。

所以 Hopper 的 SF 布局可以重新选择。下面给出推荐方案。

## 3. Hopper 推荐 SF 布局

### 3.1 L1/L2 acts SF

```
l1_acts_sf: float32 [num_max_pool_tokens, hidden / 128]    # 每 128 channel 一个 scale
            stride = {hidden/128, 1}                        # M-major（token 连续）或 K-major 都可
```

选 **M-major 还是 K-major** 取决于 TMA 加载模式：
- **K-major（推荐）**：TMA 把 `[BLOCK_M, BLOCK_K/128]` 一整块 load 到 smem，一个 K block 用完一次的 SF 是连续的。这与 [sm90_fp8_gemm_1d1d.cuh:108-111](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh#L108-L111) 的习惯一致。
- M-major 会让 SF 的读取跨 SM/warp，不如 K-major。

Dispatch pull 阶段写 `l1_acts_sf` 的代码也相应简化：从当前 SM100 的
```cpp
for each sf_uint32:
    l1_sf[sf_k * stride + transform_sf_token_idx(token_in_expert)] = remote_sf[sf_k]
```
改成
```cpp
for each sf_k:
    l1_sf[pool_token_idx][sf_k] = remote_sf[sf_k]
```

### 3.2 权重 SF

```
l1_weights_sf: float32 [num_experts_per_rank, N, K / 128]
               K-major, per expert 连续
```

权重 SF 的存储空间从 UE8M0 的 1 byte/scale 升到 float 的 4 byte/scale，但 SF 粒度从 32 改到 128 后，scale 的数量缩小 4×，所以**总字节数持平**，带宽也持平。

如果想保留原 UE8M0、32-channel 布局以节省 SF 的 HBM 存储，Hopper 可以在 TMA load 到 smem 后、promotion 前做一次软件 decode（`float(bit_hack)` 或 `exp2f`），代价是一次额外的 shared memory → FP32 转换。推荐先不优化这一步。

### 3.3 SF 的 TMA descriptor

SM100 的 SF TMA descriptor 没有 swizzle（`swizzle=0`），也没有 multicast；TMA 把 SF 拷到 smem 后，UTCCP 再拷到 TMEM。Hopper 保留 TMA descriptor 的 `swizzle=0`，去掉后续 UTCCP 步骤：

```cpp
// SM100
tma_load_sf(smem_sfa, ...);
utccp_copy(smem_sfa -> tmem_sfa);
umma(..., tmem_sfa, tmem_sfb);

// Hopper
tma_load_sf(smem_sfa, ...);
// 无 utccp
wgmma(...);
float sfa = ld_shared(smem_sfa + m_offset);  // per thread
final_accum += sfa * sfb * accum;
```

## 4. 三种 SF 的生命周期对比

```
SM100 SF 生命周期:

  GMEM/SymMem -> TMA -> smem -> UTCCP -> TMEM -> UMMA (硬件消费)

Hopper SF 生命周期:

  GMEM/SymMem -> TMA -> smem -> ld.shared -> register -> promotion (软件消费)
```

每一步都有小差异但不复杂，**最大的影响还是在布局简化**：Hopper 不需要 `transform_sf_token_idx`、不需要 M 维 128 对齐、不需要 4×32 转置。

## 5. Workspace 中 SF 相关字段的删减

[layout/mega_moe.cuh](deep_gemm/include/deep_gemm/layout/mega_moe.cuh) 中 SF 相关：

```cpp
// 可删除
num_padded_sf_pool_tokens   // 不再需要 128-align
SF_BLOCK_M = align(BLOCK_M, 128)  // 不再需要

// 保留但重定义
l1_acts_sf: float32 [num_max_pool_tokens, hidden/128]
l2_acts_sf: float32 [num_max_pool_tokens, intermediate_hidden/128]
```

symmetric buffer 的总大小会**减小**（SF 不再 padding，SF dtype 取决于方案）。

## 6. L1 Epilogue 写 L2 SF 的逻辑变化

SM100 L1 epilogue 有一步 [§10.3](reports/mega_moe_code_analysis.md#103-fp8-量化和-tma-store)：
```cpp
// 11. 写 SF 到 l2_sf_buffer (UE8M0 格式, MN-major UTCCP 布局)
//     只有 warp_idx_in_wg % 2 == 0 且 lane_idx < 4 的线程写 SF
```
这是受 UTCCP 4×32 布局约束的 stealth 写入模式。Hopper 上：
```cpp
// Hopper L1 epilogue 写 SF
// 每个 BLOCK_M 行、每 128-channel 一个 float scale
// 参与写入的线程数 = BLOCK_M 行数（每行一个 thread 写 SF）
if (thread 对应的是本行的 SF writer):
    l2_sf_buffer[pool_token_idx][sf_k_idx] = fp32_scale;
```

好处：写 SF 的代码简单很多，每行一个 writer，无需复杂映射。

## 7. 对 Dispatch 拉取的影响

Dispatch 拉 SF 的代码原本在 [§8.7](reports/mega_moe_code_analysis.md#87-拉取远端-token-到本地-l1-pool)：
```
for each sf_uint32:
    l1_sf[sf_k * stride + transform_sf_token_idx(token_in_expert)] = remote_sf[sf_k]
```
Hopper 改为（如果保留 UE8M0 的 32-channel 原始 SF，则 sf_k 范围 hidden/32；如果重新量化为 128-channel float，则 hidden/128）：
```
for each sf_k:
    l1_sf[pool_token_idx][sf_k] = remote_sf[sf_k]
```

注意这里需要调用方（Python）保证输入 `x_sf` 的 dtype 和 layout 与 kernel 约定匹配：
- 如果 kernel 用 float per-128-channel，`x_sf` 也是 float per-128-channel，`per_token_cast_to_fp8(..., use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)`。
- 如果保持 UE8M0 per-32-channel，则激活量化不变，kernel 做 decode。

**建议**：**统一用 float per-128-channel 方案**，和 [sm90_fp8_gemm_1d1d](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh) 对齐，减少特例。

## 8. 小结

- UTCCP 消失，SF 的硬件对齐约束全部解除。
- 推荐 Hopper 方案：SF 用 **float per-128-channel K-major**，与 DeepGEMM FP8 GEMM 惯例一致。
- `_transpose_sf_for_utccp` Python 函数、`transform_sf_token_idx` kernel lambda、`num_padded_sf_pool_tokens` workspace 字段、`SF_BLOCK_M` 常量 → 全部删除。
- Dispatch pull 和 L1 epilogue 写 SF 的代码大幅简化。
- 激活侧 `x_sf` 的 Python 生成也需要改（`use_ue8m0=False, gran_k=128`）。
