# 04 · FP4 权重：Hopper 没有原生 FP4 MMA 的方案选择

> **难点强度：★★★☆☆**（主要是工程决策而非硬件黑魔法）

## 1. 背景

SM100 Mega MoE 的权重精度是 **FP4 E2M1**，激活是 **FP8 E4M3**，通过 `SM100_MMA_MXF8F6F4_2x1SM_SS` 指令直接消费。Hopper 没有任何 FP4/FP6 MMA，只支持 FP8 E4M3/E5M2、FP16、BF16、TF32 的 WGMMA。

因此移植到 Hopper 必须在三个方案中选一个：

| 方案 | 权重存储 | GMEM 带宽 | HBM 容量 | MMA 精度 |
|---|---|---|---|---|
| A. 权重预反量化为 FP8 | FP8 E4M3 | 2× 于 FP4 | 2× | FP8 × FP8（原生）|
| B. 权重仍以 FP4 存储，kernel 内即时反量化 | FP4 packed | 1× 于 SM100 | 1× | FP8 × FP8（软件反量化）|
| C. 权重以 BF16 存储 | BF16 | 4× | 4× | BF16 × BF16 |

## 2. 方案分析

### 2.1 方案 A：预反量化为 FP8（**推荐起步方案**）

Python 侧 `transform_weights_for_mega_moe` 在离线阶段做 `fp4 → fp8` 转换：
```python
def transform_weights_for_mega_moe_sm90(l1_w, l2_w, l1_sf, l2_sf):
    # 反量化 FP4 + UE8M0 scale → 拿到 FP32 权重
    # 再按 per-128-channel 重新量化为 FP8 E4M3 + 新 scale（float32）
    l1_w_fp8, l1_sf_new = fp4_to_fp8_with_channel_scale(l1_w, l1_sf, gran_k=128)
    l2_w_fp8, l2_sf_new = fp4_to_fp8_with_channel_scale(l2_w, l2_sf, gran_k=128)
    # 去掉 gate/up interleave（见 02 章）
    return l1_w_fp8, l2_w_fp8, l1_sf_new, l2_sf_new
```

**优点**：
- 完全避开 Hopper 不支持 FP4 的问题。
- kernel 里 MMA 流程和 DeepGEMM 现有的 [sm90_fp8_gemm_1d1d.cuh](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh) 几乎一致。
- 权重 SF 的 layout 可以直接用 DeepGEMM 现有的「per-128-channel, MN-major」，利用 TMA 加载到 smem 后 per-thread `ld.shared` 消费（简单 / 兼容 DeepGEMM 生态）。

**缺点**：
- HBM 占用 **2×**：MoE 权重往往是模型总权重的大头，256 expert × 2 GEMM × shape 级别的扩张非常昂贵。例如 256 个 expert、inter=5120、hidden=5120、FP4 ~ 6.7 GB，FP8 → 13.4 GB，对 H100 80 GB 而言每个 rank 增加 2-3% 容量，对 H200 141 GB 可以接受。
- HBM 读取带宽 **2×**：L1/L2 GEMM 的 tensor core 利用率本来就偏低（MoE token 不均、小 batch），带宽翻倍后瓶颈从 compute-bound 进一步退到 memory-bound。

**适用场景**：中等 batch size、追求最低实现复杂度、H200 或 B 以上容量充足的机器。

### 2.2 方案 B：kernel 内即时反量化（FP4 → FP8）

把 FP4 权重在 TMA load 到 shared memory 时仍保持 packed 形态（2 × FP4 = 1 × uint8），由 consumer 侧在每次 WGMMA 之前把 shared memory 里的 FP4 反量化为 FP8 写回另一个 shared memory buffer（或者寄存器），再发射 WGMMA。

**优点**：
- HBM 容量和带宽跟 FP4 持平（1×），这是最有吸引力的点。
- 可以和方案 A 复用同一套权重离线表达（就是原 SM100 的 FP4 权重）。

**缺点**：
- **反量化代价**：每 K=32 的 WGMMA 子 block 需要 BLOCK_N=128 个 FP4 值反量化成 FP8，加上乘以 UE8M0 scale，每次 128 个乘加 + 4 次 2-bit 展开。对整个 L1 GEMM（N×K×M）累计在 hundred of millions 级别的额外 ALU 指令。
- **Shared memory 翻倍**：反量化后需要一个 FP8 buffer 供 WGMMA 使用，原本的 FP4 packed buffer 还占着（producer 写入），两份共存 → smem stage 减少 1 个。
- **与 WGMMA 的 RS (register-source) 变体配合**：若把反量化后的 FP8 留在寄存器里，需要用 `wgmma.mma_async.sync.aligned ... RS` 形态，但 RS 只支持 A 从寄存器、B 从 smem。由于 Mega MoE 做了 A/B swap，**权重在 A**，刚好能用 RS；但 RS 又要占 warpgroup 的寄存器（每 thread 每 WGMMA iteration 至少 16 个寄存器装 FP8），再次挤压 epilogue register budget。
- **UE8M0 scale 直接乘在 FP8 值上**会损失精度（UE8M0 是 2 的幂次，本来是乘在累加后；这里乘在反量化的 FP8 值上意味着 FP8 值会饱和到 448 的情况增多）。要么升级到 float scale，要么改回方案 A。

**适用场景**：带宽严重受限、权重是热点、愿意付出可观的实现复杂度。

### 2.3 方案 C：BF16 权重

**优点**：
- 实现最简单：现成的 `sm90_bf16_gemm.cuh` 可以借。
- 不需要 SF pipeline（05 章的 block scaling 都省了）。

**缺点**：
- HBM 4×、Tensor core 吞吐只有 FP8 的一半（`m64n128k16` 对 BF16 而言 K 只有 16，FP8 是 K=32）。
- 对 MoE 基本不现实（容量 × 性能都劣化）。

## 3. 推荐方案：A 作为 v0，B 作为 v1

建议分两步：

```
v0 (正确性优先):
  1. 离线把 FP4 权重 → FP8 权重（方案 A）
  2. 基于 sm90_fp8_gemm_1d1d.cuh 的 MMA + 05 章 CUDA promotion
  3. 验证 fused kernel 输出与 baseline bitwise（无 FP4 重量化带来的误差差异较大，bitwise 不等，需要改成 tolerance 比对）

v1 (带宽优化):
  4. 在 v0 正确后，尝试 kernel 内即时反量化（方案 B）
  5. 需要对比 TFLOPS / HBM GB/s，选择性能最优点
```

## 4. 权重转换代码草图（方案 A）

```python
# deep_gemm/mega/__init__.py 新增
def transform_weights_for_mega_moe_sm90(l1_weights_fp4, l2_weights_fp4,
                                         l1_sf_ue8m0, l2_sf_ue8m0):
    # 1. Dequant FP4 → FP32，使用原 UE8M0 scale（per 32 channel）
    l1_fp32 = dequant_fp4_ue8m0(l1_weights_fp4, l1_sf_ue8m0, gran_k=32)
    l2_fp32 = dequant_fp4_ue8m0(l2_weights_fp4, l2_sf_ue8m0, gran_k=32)

    # 2. Re-quant FP32 → FP8 E4M3，per-128-channel float scale
    l1_fp8, l1_sf = per_channel_cast_to_fp8(l1_fp32, gran_k=128)
    l2_fp8, l2_sf = per_channel_cast_to_fp8(l2_fp32, gran_k=128)

    # 3. 不做 gate/up interleave（Hopper WGMMA 分布不支持 SM100 的交错）
    # 4. SF layout 改为 sm90_fp8_gemm 规范 (K-major 或 MN-major 依 DeepGEMM 惯例)
    return l1_fp8, l2_fp8, l1_sf, l2_sf
```

关键变动：
- `gran_k` 从 32（FP4 native）改成 128（FP8 per-channel scaling，与 DeepGEMM FP8 规范一致）。
- 不再做 `_interleave_l1_weights` 的 gate/up 交错（因为 Hopper 累加器 layout 不匹配，见 02.3.3）。
- 不再调用 `_transpose_sf_for_utccp`（UTCCP 不存在，见 07 章）。

## 5. HBM 带宽敏感度估算

以一个 EP8 DeepSeek 配置（num_experts=256, top_k=6, hidden=7168, inter=2048）做估算：

| 权重表示 | L1 + L2 权重 GB | 一次 forward 的权重读 | 相对 FP4 |
|---|---|---|---|
| FP4 (原 SM100) | ~6.0 GB | 6.0 GB | 1.0× |
| FP8（方案 A） | ~12.0 GB | 12.0 GB | 2.0× |
| BF16（方案 C） | ~24.0 GB | 24.0 GB | 4.0× |

H100 HBM3 带宽 3 TB/s，读 12 GB 约 4 ms（纯权重读，未算激活/SF/overhead）。对应 token latency 在 bsz=64, seq=128 下已经足够成为瓶颈。这是方案 A 的主要风险点，也是后续考虑方案 B 的动机。

## 6. 小结

- Hopper 无 FP4，必须在 FP8（A）、即时反量化（B）、BF16（C）间选。
- 推荐先走 **方案 A** 离线反量化成 FP8，最大限度复用 DeepGEMM sm90 生态，快速把 kernel 调通。
- 长期看 **方案 B** 有 2× HBM 带宽优势，但实现复杂度很高、对寄存器预算侵入大。
- 方案 C 不推荐。
- 离线权重转换脚本需要同时改三件事：反量化、去交错、SF layout 适配。
