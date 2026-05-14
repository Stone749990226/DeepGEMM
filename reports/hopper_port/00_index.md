# Mega MoE 移植到 Hopper (FP8) 难点分析 · 索引

本系列报告分析 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` 从 Blackwell SM100 移植到 Hopper SM90 的难点，精度基线定为 **FP8 E4M3 激活 / FP8 E4M3 权重**（放弃 FP4）。

系列共 12 章，按「硬件能力缺失 → 软件架构重构 → 工程路线图」顺序组织。每章自包含，可独立审阅。

## 章节

| # | 标题 | 难点强度 | 摘要 |
|---|---|---|---|
| [01](01_overview_and_hw_gap.md) | 总览与 SM100→SM90 硬件差距 | — | 给出差距矩阵和移植工作量估算 |
| [02](02_tmem_to_register.md) | Tensor Memory → 寄存器累加器 | ★★★★★ | TMEM 消失，累加器搬进寄存器，gate/up 交错特性失效 |
| [03](03_umma_vs_wgmma.md) | UMMA → WGMMA 语义差异 | ★★★★☆ | 发射粒度、累加器位置、完成通知、N 动态性全面变化 |
| [04](04_fp4_weight_choices.md) | FP4 权重方案选择 | ★★★☆☆ | 离线 FP4→FP8（A）/ 即时反量化（B）/ BF16（C）三选一 |
| [05](05_block_scaling_rewrite.md) | Block-scaled MMA 缺失 → CUDA promotion | ★★★★☆ | SF 从硬件消费改为软件 `final_accum += SFA×SFB×accum` |
| [06](06_cluster_multicast_rework.md) | 2-CTA MMA / cluster multicast 重构 | ★★★★☆ | Hopper 无 2-CTA MMA，只能靠 TMA multicast 共享 A tile |
| [07](07_sf_layout_without_utccp.md) | UTCCP 缺失后 SF 布局重构 | ★★★☆☆ | 去掉 4×32 转置和 128 对齐，SF 改为 float per-128-channel |
| [08](08_epilogue_swiglu_rework.md) | SwiGLU Epilogue 重构 | ★★★★☆ | gate/up 配对策略、amax reduction、FP8 量化全部在 register 域重做 |
| [09](09_smem_budget_pipeline.md) | Shared memory 预算与 pipeline 深度 | ★★★☆☆ | 权重 2× 导致 stage 从 12 掉到 6，cluster multicast 恢复 |
| [10](10_comm_sync_portability.md) | symm mem + NVLink barrier 的 Hopper 适配 | ★★☆☆☆ | 基本无需改动，主要是拓扑验证 |
| [11](11_heuristic_and_numerics.md) | Heuristic 重标定与数值精度 | ★★★☆☆ | BLOCK_M 候选改 `{16,32,64,128}`，bitwise → tolerance 比对 |
| [12](12_porting_roadmap.md) | 分阶段移植路线图 | — | 6 个 phase，v0+v1 大约 3-4 周 |

## 建议阅读顺序

**需要整体把握**：01 → 12 → 02 → 08 → 其余按需
**实现者视角**：01 → 02 → 03 → 05 → 08 → 09 → 12
**决策者视角**：01 → 04 → 11 → 12

## 核心结论

1. **移植核心难度集中在 MMA + Epilogue + SF pipeline**（约 800 行重写），上层框架（host API、workspace、scheduler、dispatch、combine）基本可保留。
2. **每个硬件缺失都在软件层级联反应**：TMEM → 寄存器压力 → BLOCK_M 上限 → pipeline stage → 需要 multicast 恢复 → scheduler 约束；UTCCP → SF 布局简化 → CUDA promotion → 寄存器再压缩；FP4 → FP8 → HBM 带宽 2× → stage 数更紧。
3. **推荐分阶段路线**：v0 单 CTA + 离线 FP4→FP8（方案 A） → v1 cluster multicast + heuristic 标定 → v2 warp-specialized epilogue（可选）。
4. **预期 Hopper 版本吞吐为 SM100 的 40-50%**，相对 legacy 分阶段实现有 1.2-2× 加速，有明确的工程价值。
