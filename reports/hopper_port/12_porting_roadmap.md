# 12 · 分阶段移植路线图

> 本章是整个系列的收官，把 01-11 章的分析落到 **可执行的阶段化计划**。

## 1. 移植的关键决策回顾

| 决策点 | 选择 | 章节 |
|---|---|---|
| 目标架构 | SM90 (H100 / H200) | 01 |
| 权重精度 | FP8 E4M3（方案 A：离线 FP4→FP8） | 04 |
| MMA 指令 | `wgmma.mma_async m64nNk32 F32E4M3E4M3` | 03 |
| 累加器位置 | Warpgroup 寄存器 | 02 |
| SF 方案 | CUDA promotion, float per-128-channel | 05, 07 |
| Cluster 策略 | v0 单 CTA，v1 cluster=2 + multicast A | 06, 09 |
| BLOCK_M 候选 | `{16, 32, 64, 128}` | 11 |
| BLOCK_N | 128（gate+up 合并逻辑 N） | 08, 09 |
| BLOCK_K | 128（K=32 MMA × 4 次）| 09 |
| Warp 结构 | v0：2 warpgroup（producer+consumer），v2：3 warpgroup | 02, 09 |
| SwiGLU gate/up 配对 | N 维连续两半切分（策略 2）| 08 |
| Dispatch/Combine | 基本不变，保留 NVLink pull + symm mem | 10 |
| Heuristic | 重新标定，阈值扫表 | 11 |

## 2. 文件清单与改动规模

```
新增 / 重写:
  deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh   ★★★★★  ~1200 行
  csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp              ★★★★☆  ~400 行
  csrc/jit_kernels/heuristics/sm90_mega_moe.hpp             ★★★☆☆  ~250 行

修改（加 arch dispatch 分支）:
  csrc/apis/mega.hpp                                         ★★☆☆☆  ~40 行
  deep_gemm/mega/__init__.py                                 ★★★☆☆  ~80 行
  deep_gemm/include/deep_gemm/layout/mega_moe.cuh            ★★☆☆☆  ~60 行
  deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh         ★★☆☆☆  ~30 行
  tests/test_mega_moe.py                                     ★★★☆☆  ~100 行
```

## 3. 阶段划分

### Phase 0 · Pre-flight（1-2 天）

**目标**：确认 Hopper 子系统可用，不动 Mega MoE 代码。

- [ ] PyTorch symm memory 在 H100 多卡上跑通（附录 A.1 小测）
- [ ] 跨 rank TMA load 的 smoke test
- [ ] DeepGEMM 现有 `sm90_fp8_gemm_1d1d` 在当前环境编译通过并跑测试
- [ ] 确认 gcc/nvcc 版本（CUDA ≥ 12.4 + PyTorch ≥ 2.9）

**出口**：所有子系统绿灯，可以进入 kernel 编写。

### Phase 1 · 权重转换与 Host API（2-3 天）

**目标**：离线权重转换 + Python API + C++ dispatch 架构搭好，但**kernel 可以是空壳 / 直接返回 zero**。

- [ ] 实现 `transform_weights_for_mega_moe_sm90(...)`：FP4 dequant + FP8 requant + SF 重建（04 章）
- [ ] `get_symm_buffer_for_mega_moe` 的 SM90 变体（SF 不再 128-align，pool 容量重算）
- [ ] `csrc/apis/mega.hpp` 的 `fp8_mega_moe` 入口（`arch_major == 9` 分支）
- [ ] `csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp` 框架（TMA descriptor 创建 + JIT 代码模板字符串）
- [ ] Kernel 用 dummy 实现（launch 成功但不算结果）
- [ ] 能用 `tests/test_mega_moe.py` 的 Python 调用链 launch kernel，验证 host 侧不崩

**出口**：kernel 能 launch，Python API 完整，工作重心转到 device code。

### Phase 2 · L1 GEMM Kernel（MVP，5-7 天）

**目标**：跑通 L1 phase 的 WGMMA + CUDA promotion，输出正确的 FP8 结果。

- [ ] Warpgroup 结构：`warpgroup_reg_alloc` producer/consumer 分配
- [ ] Scheduler 移植（去 cluster 约束的简化版本）
- [ ] Dispatch pull 完整移植（符号内存写 src_index、pull remote token）
- [ ] TMA load A/B/SFA/SFB 走 mbarrier full/empty pipeline
- [ ] WGMMA issue + `wait_group<0>` + CUDA promotion
- [ ] L1 epilogue：SwiGLU + amax + FP8 cast + 写 smem（不做 TMA store，先验证 accum 正确）

**中期 milestone**：能把 dispatch 后的激活和单一 expert 的权重算出 L1 FP32 结果，和 PyTorch reference matmul `assert_close(atol=1e-3)` 通过。

**出口**：L1 GEMM 输出正确，但不一定高性能。

### Phase 3 · L1 Epilogue 完整 + L2 GEMM（5-7 天）

**目标**：完整跑通 L1→L2 的 fused chain，最终 y 输出（通过 combine reduce）。

- [ ] L1 epilogue：TMA store 到 l2_acts + 写 SF + 设置 l2_arrival_mask bit
- [ ] L2 GEMM：等 l2_arrival_mask → TMA load → WGMMA + promotion
- [ ] L2 epilogue：BF16 cast + NVLink write-back 到 combine buffer
- [ ] Final combine reduce：TMA load combine_buffer + FP32 reduce + TMA store y
- [ ] Workspace 清理

**中期 milestone**：`tests/test_mega_moe.py` 在 H100 上跑出接近 baseline 的输出（`assert_close(atol=1e-2)`）。

**出口**：端到端 fused kernel 通过数值测试。

### Phase 4 · 正确性稳定化（2-3 天）

- [ ] 所有测试矩阵（11 章 §2.3）跑通
- [ ] 多 rank 测试：EP4 / EP8 / EP16（不同 num_experts/ranks 组合）
- [ ] Mask ratio 非 0 的路由不均衡测试
- [ ] Stress test：连续 100 次 kernel 调用，验证 workspace 清理正确、无状态泄露
- [ ] 内存泄漏/越界检查（compute-sanitizer）

**出口**：Hopper Mega MoE v0 可合并。

### Phase 5 · 性能优化 v1（5-7 天）

**目标**：启用 cluster + multicast，恢复 pipeline stage 数。

- [ ] 启用 cluster=2，TMA multicast A tile
- [ ] Scheduler 添加 2-block M 约束，两个 CTA 对称运行
- [ ] 重新调 heuristic（11 章）：BLOCK_M 阈值扫表
- [ ] 寄存器使用 profiling，微调 `setmaxnreg`

**出口**：TFLOPS 达到 SM100 版本的 40-50%。

### Phase 6 · 性能优化 v2（可选，1-2 周）

**目标**：warp-specialized epilogue，进一步提高 overlap。

- [ ] 切 3 warpgroup：producer / math / epilogue
- [ ] Math warpgroup 将 accum 通过 smem 传给 epilogue warpgroup
- [ ] Ping-pong 双 math warpgroup（寄存器允许时）
- [ ] Profile-guided 优化

**出口**：性能接近 Hopper 该 workload 的理论上限。

## 4. 关键风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| WGMMA 寄存器分布不配合 SwiGLU gate/up 配对 | 中 | 高 | 策略 2 N 维两半切分（08 章）|
| Pipeline stage 降到 < 4 性能退化 | 高 | 中 | Phase 5 启用 cluster multicast（09 章）|
| 数值精度与 legacy gap 过大（>1%） | 中 | 中 | 检查 SF 粒度、clamp、量化顺序（11 章）|
| 符号内存 + TMA 跨 rank 在 H100 PCIe 机器失效 | 低 | 高 | Phase 0 验证拓扑，文档化 NVLink 全连接需求（10 章）|
| FP4→FP8 权重转换精度损失过大 | 中 | 中 | 保留 FP4 SF 的精度；允许 reference 用同样转换的权重（04 章）|
| Hopper setmaxnreg 不够，BLOCK_M=128 超寄存器 | 中 | 中 | 降级到 BLOCK_M=64；或 v2 的 3-warpgroup 结构（02 / 08 章）|

## 5. 总工作量估计

| Phase | 天数（单人工程师）|
|---|---|
| 0 · Pre-flight | 1-2 |
| 1 · Host API | 2-3 |
| 2 · L1 Kernel MVP | 5-7 |
| 3 · L1→L2 完整 | 5-7 |
| 4 · 稳定化 | 2-3 |
| 5 · 性能 v1 | 5-7 |
| 6 · 性能 v2（可选）| 10-14 |
| **合计（v0+v1）** | **~3-4 周** |
| **合计（含 v2）** | **~5-6 周** |

## 6. 移植 vs 从头写的取舍

有一种替代方案是**不移植现有 Mega MoE，而是从头基于 DeepGEMM sm90 FP8 GEMM + DeepEP + TileLang SwiGLU 组合出一个 Hopper 版 fused kernel**。对比：

| 维度 | 移植 SM100 Mega MoE | 从 Hopper 生态拼装 |
|---|---|---|
| 总代码量 | 较少（复用 dispatch/combine/scheduler）| 较多 |
| 架构对齐度 | 低（强改）| 高（自然适配 Hopper）|
| 维护复杂度 | 两套 kernel 并行 | 一套 |
| 时间 | 3-6 周 | 4-8 周 |
| 与 SM100 行为一致性 | 强（同一份语义）| 弱（可能分歧）|

**建议**：**采用移植路线**，理由：
1. Mega MoE 的 dispatch / combine / scheduler / NVLink pipeline 是 Hopper 同样需要的，复用成本最低。
2. 维持 Python API 一致性（同一个 `fp8_mega_moe` 函数名，arch-dispatch），用户无感。
3. SM100 已经验证了整体算法的正确性，Hopper port 只在 MMA/epilogue 层发力，bug 面积小。

## 7. 可交付清单

```
reports/hopper_port/
├── 01_overview_and_hw_gap.md           (本系列 01)
├── 02_tmem_to_register.md              (本系列 02)
├── 03_umma_vs_wgmma.md
├── 04_fp4_weight_choices.md
├── 05_block_scaling_rewrite.md
├── 06_cluster_multicast_rework.md
├── 07_sf_layout_without_utccp.md
├── 08_epilogue_swiglu_rework.md
├── 09_smem_budget_pipeline.md
├── 10_comm_sync_portability.md
├── 11_heuristic_and_numerics.md
└── 12_porting_roadmap.md                (本文)
```

每份文档都是自包含的技术备忘录，可独立被 reviewer 审阅。

## 8. 一句话总结

Mega MoE 移植到 Hopper FP8 的**难点不在于任何单一硬件缺失，而在于每一块缺失都要用软件路径重构一次，并且这些重构彼此耦合**：TMEM 没了 → 累加器进寄存器 → BLOCK_M 受限 → pipeline stage 下降 → 需要 cluster multicast 恢复 → 又要求 scheduler 保持偶数 block 约束；UTCCP 没了 → SF 布局简化 → Python 权重 transform 重写 → kernel 读 SF 改成 ld.shared → CUDA promotion 引入新的寄存器 → 再次压缩 epilogue 预算；FP4 没了 → 权重转 FP8 → HBM 带宽 2× → pipeline stage 更吃紧……

整体复杂度是 **模块级重写**（MMA + Epilogue + SF pipeline ≈ 800 行），但保留了 host API、workspace、scheduler 状态机、dispatch pull、combine reduce 的上层骨架。按路线图分 6 个 phase 推进，v0+v1 大约 3-4 周可以拿到一个可用、与 legacy 相比有显著加速的 Hopper Mega MoE FP8 kernel。
