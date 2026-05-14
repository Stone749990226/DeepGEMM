#  01 · 总览与 SM100→SM90 硬件差距

> 目标：把 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` 所实现的 Mega MoE fused kernel，从 Blackwell (SM100, compute capability 10.0) 移植到 Hopper (SM90, H100/H200)，精度基线维持 **FP8 E4M3 激活 / FP8 E4M3 权重**（放弃 FP4 权重，见 04）。

本系列报告把移植难点按「硬件能力缺失」和「软件架构重构」两条线索分 12 章详述。本章先建立整体地图。

---

## 1. 一句话结论

当前 Mega MoE 是一个**高度 SM100 专属**的 kernel：它的指令、内存层级、线程编排、scale factor 流水线几乎每一层都用到了 SM100 的独有硬件。移植到 Hopper 不是简单的 `#if __CUDA_ARCH__` 分支，而是**相当于重写 MMA + epilogue 的 80%**，保留的只是：

- Host 侧 Python/C++ API 框架（`mega/__init__.py`、`csrc/apis/mega.hpp`）
- symmetric memory 布局与 workspace 元数据（`layout/mega_moe.cuh`）
- Device scheduler 的 wave 状态机（`scheduler/mega_moe.cuh`）
- Dispatch 阶段的 NVLink pull + round-robin min-peeling 算法
- Final combine reduce 的 TMA load/store 双缓冲

需要重写的部分：
- L1/L2 GEMM 的 MMA 指令、累加器存储、SF pipeline、cluster 编排
- L1/L2 Epilogue 的 TMEM load、SwiGLU 配对、FP8 量化
- Shared memory 预算、stage 数、寄存器分配
- Heuristic 的 BLOCK_M 候选集与 cost model

---

## 2. SM100 → SM90 硬件差距矩阵

下表列出移植时直接相关的硬件特性差异，每一项都在后续章节展开。

| 特性 | SM100 (Blackwell) | SM90 (Hopper) | 影响模块 | 章节 |
|---|---|---|---|---|
| Tensor Memory (TMEM) | 独立的 256KB 片上累加器，`cute::TMEM::Allocator2Sm` | **不存在** | MMA 累加器，SF 暂存 | 02 |
| MMA 指令 | UMMA (`SM100_MMA_MXF8F6F4_2x1SM_SS`) 2-CTA shared→shared | WGMMA (`wgmma.mma_async.sync.aligned`) 单 warpgroup | L1/L2 GEMM 核心 | 03 |
| FP4 E2M1 原生 MMA | ✅ UMMA 支持 MXF8F6F4 混合 | ❌ 只有 FP8/FP16/BF16/TF32 MMA | 权重精度选择 | 04 |
| Block-scaled MMA | ✅ 硬件直接消费 SFA/SFB (UE8M0) | ❌ 必须 CUDA 端手动 promotion (`scale_a × scale_b × accum`) | SF pipeline、寄存器压力 | 05 |
| 2-CTA cluster MMA (multicast A) | ✅ `2x1SM_SS` 指令两 CTA 共享 A | ❌ WGMMA 是单 warpgroup；cluster 只能靠 TMA multicast | A tile 复用策略 | 06 |
| UTCCP (Unified TMEM CP) | ✅ `SM100_UTCCP_4x32dp128bit_2cta`：smem→TMEM SF 拷贝 | ❌ SF 保持在 shared memory，per-thread `ld.shared` | SF 物理布局、interleave | 07 |
| TMEM Load 指令 | ✅ `SM100_TMEM_LOAD_16dp256b1x` 一次读 8 FP32 | ❌ 累加器本就在寄存器，无需 load | L1 epilogue 取 accumulator 的路径 | 02 / 08 |
| Shared memory 容量 | 232,448 B / SM (用户可见) | 228 KB / SM | pipeline stage 数 | 09 |
| 寄存器文件 | 64K × 4B / SM | 64K × 4B / SM（不变） | 累加器吃的 register 数翻倍 | 02 / 09 |
| `cluster.arrive.relaxed` + `fence_barrier_init` | ✅ | ✅（Hopper 同样支持）| cluster 初始化 | 06 |
| mbarrier / TMA 流水线 | ✅ | ✅（Hopper 引入的，语义一致）| TMA producer / consumer | 不变 |
| Symmetric memory (IMEX) | ✅ | ✅（H100 支持 NVLink p2p + CUDA 13 symm mem）| dispatch/combine | 10 |
| `red.async.relaxed.gpu` 等系统范围 atomic | ✅ | ⚠ 语义存在，性能不同 | NVLink barrier | 10 |

---

## 3. Mega MoE kernel 中 SM100 专属符号的全量清单

> 通过 `grep` 从 `sm100_fp8_fp4_mega_moe.cuh` 直接抽出来的硬件依赖点，按「必须重写 / 可保留」标注。

**必须重写（硬件指令不存在）**
- `cute::TMEM::Allocator2Sm`（TMEM 分配器，line 65）
- `Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem)`（line 309）
- `cute::SM100_UTCCP_4x32dp128bit_2cta`（line 840：smem→TMEM SF 拷贝）
- `ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma`（line 863：2-CTA block-scaled MMA）
- `cute::SM100_TMEM_LOAD_16dp256b1x::copy`（line 981/983/1145/1147：从 TMEM 读累加器）
- `cute::UMMA::make_instr_desc_block_scaled<...>`（line 781：UMMA 描述符）
- `mma::sm100::make_umma_desc<...>`（line 789/790：UMMA A/B 描述符）
- `umma_arrive`（line 825：UMMA 完成后到达 TMEM barrier）
- `cutlass::arch::fence_view_async_tmem_load`（line 985/1149）
- `tmem_full_barriers / tmem_empty_barriers`（line 267/268：TMEM 生产-消费 barrier）
- `kNumAccumTmemCols`（line 217：TMEM 列数分配）

**可保留或小改**
- `ClusterTransactionBarrier` / mbarrier —— SM90 同样支持
- `cute::prefetch_tma_descriptor` —— 同样支持
- `ptx::sync_aligned / sync_unaligned` —— named barrier 通用
- `comm::grid_sync` —— 基于 atomic，任意 SM 都可用
- `comm::nvlink_barrier` —— 基于 symm mem + system-scope atomic，H100 支持

---

## 4. 为什么 Hopper 的 FP8 版本「仍然值得」

FP4 是 Blackwell 的新增能力，Hopper 只能退回 FP8。但这不等于移植没意义：

1. **工程价值**：H100 / H200 是当前部署主力，DeepSeek / Qwen 等 MoE 推理大量跑在 H100 上。Mega MoE 融合 dispatch/GEMM/combine 的思路对 Hopper 依然解决真实问题（kernel launch 开销、NVLink 占空比低）。
2. **理论带宽利用**：H100 的 FP8 Tensor Core 吞吐是 2 PFLOPS (sparse) / 1 PFLOPS (dense)，NVLink 900 GB/s；在 EP 小 batch 场景下，计算远未打满，瓶颈在通信 + launch，fused kernel 的收益**不依赖 FP4**。
3. **可对比验证**：DeepGEMM 已经有成熟的 `sm90_fp8_gemm_1d1d.cuh`，其 FP8 + CUDA promotion 的 scale 流水线可以直接借用 ([deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:246-308](deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh#L246-L308))，是非常好的参考。

---

## 5. 移植工作量估算

| 模块 | SM100 源码行数（估） | SM90 移植行数（估） | 工作量 |
|---|---|---|---|
| Host API / 校验 / 权重 transform | ~400 | ~350 | 小（去 FP4 逻辑）|
| Workspace + Buffer layout | ~600 | ~550 | 小（pool 容量重算）|
| Scheduler 状态机 | ~300 | ~280 | 小（cluster 约束可能放宽）|
| **MMA / Epilogue 核心** | **~900** | **~800** | **大（全重写）**|
| Dispatch 跨 rank pull | ~200 | ~200 | 基本不变 |
| Final combine reduce | ~150 | ~150 | 基本不变 |
| Heuristic / JIT | ~300 | ~300 | 需全部重新标定 |
| 测试 / baseline | ~400 | ~300 | 去掉 FP4 路径 |

核心难度集中在 **MMA + Epilogue + SF pipeline** 这 ~800 行的重构，是本系列报告的主线。

---

## 6. 后续章节导航

- **02** `02_tmem_to_register.md` — TMEM 累加器 → register file 的代价与策略
- **03** `03_umma_vs_wgmma.md` — UMMA 和 WGMMA 的指令语义差异
- **04** `04_fp4_weight_choices.md` — FP4 权重三种方案（预反量化 / 即时反量化 / 直接 FP8）
- **05** `05_block_scaling_rewrite.md` — block-scaled MMA 缺失 → CUDA promotion pipeline
- **06** `06_cluster_multicast_rework.md` — 2-CTA SS MMA 被迫退化为单 CTA WGMMA
- **07** `07_sf_layout_without_utccp.md` — UTCCP 去掉后 SF 必须重新设计物理布局
- **08** `08_epilogue_swiglu_rework.md` — SwiGLU / amax / FP8 量化 在 register 域重构
- **09** `09_smem_budget_pipeline.md` — shared memory 预算与 pipeline 深度重算
- **10** `10_comm_sync_portability.md` — symmetric memory 与 NVLink barrier 的 Hopper 适配
- **11** `11_heuristic_and_numerics.md` — Heuristic 重标定与 FP8 数值精度
- **12** `12_porting_roadmap.md` — 分阶段移植路线图
