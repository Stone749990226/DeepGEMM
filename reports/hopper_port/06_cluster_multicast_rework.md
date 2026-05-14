# 06 · 2-CTA MMA 与 Cluster / Multicast 重构

> **难点强度：★★★★☆**

## 1. SM100 Mega MoE 的 cluster 用法

当前 kernel 以 **cluster size = 2** 运行：
- 两个 CTA 构成一个 cluster，共享一次 UMMA 计算。
- `SM100_MMA_MXF8F6F4_2x1SM_SS` 指令是 **"2x1SM_SS"**：两个 SM 合作消费同一块 A tile，每个 SM 各自算一半 M（`LOAD_BLOCK_M = BLOCK_M / 2`）。
- TMA 用 **multicast**：一次 TMA load 同时把 A tile 写入两个 CTA 的 shared memory。
- Scheduler 里强制 `kNumL1BlockNs % 2 == 0 && kNumL2BlockNs % 2 == 0 && kNumSMs % 2 == 0`，保证 cluster 中的 2 个 CTA 总是落在同一个 M block，相邻的 N block。

直接受益：
- **A tile TMA 带宽减半**（广播而不是两个独立 load）。
- UMMA 的累加器在 TMEM 分布在两个 CTA，等效 M 维放大 2×。
- epilogue 只在 leader CTA 做 MMA issue，reduce 整 cluster 的 issue 开销。

## 2. Hopper 没有 2-CTA MMA

SM90 WGMMA 是 **单 warpgroup 级别**的指令，**没有 multi-CTA 协同 MMA**。cluster 在 Hopper 上仍然存在（thread block cluster 是 SM90 新引入的），但：

| Hopper cluster 功能 | 支持情况 |
|---|---|
| DSMEM（distributed shared memory，跨 CTA `ld.shared::cluster`）| ✅ 支持 |
| TMA multicast（一次 TMA load 写多个 CTA smem）| ✅ 支持（`cp.async.bulk.tensor ... .multicast`）|
| Cluster barrier（`barrier.cluster.arrive/wait`）| ✅ 支持 |
| 2-CTA MMA（两 CTA 共享一个 MMA）| ❌ 不支持 |

所以移植到 Hopper 后：
- 可以保留 cluster，用于 **TMA multicast A tile**（让两个 CTA 共享一次 GMEM→smem 的拷贝）。
- 但两个 CTA 的 MMA 是**各自独立**发射的，没办法合起来算一个大 tile。
- 不存在「leader CTA 代表整 cluster 发射」的概念 —— 每个 CTA 自己的 warpgroup 发 WGMMA。

## 3. 移植策略选择

有两种策略：

### 3.1 策略 A：保留 cluster，但只用 multicast（**推荐**）

```
Cluster size 2:
┌─────────────────────────────────────────────────────────────┐
│ CTA 0                                    CTA 1              │
│  warpgroup TMA (producer)                warpgroup TMA (prod)│
│    └── cp.async.bulk.tensor.multicast ──> 同一个 A tile      │
│                                            (smem_a 共享)    │
│  warpgroup Math (WGMMA 独立)             warpgroup Math     │
│    - m_block = scheduler.m                 - m_block = sched.m │
│    - n_block = scheduler.n + 0             - n_block = sched.n + 1 │
└─────────────────────────────────────────────────────────────┘
```

- 两个 CTA 落在**同一个 M block，不同 N block**（这和 SM100 当前 scheduler 约定一致）。
- A tile（激活 token）被两个 CTA 共享，TMA multicast 一次就够。
- B tile（权重）每个 CTA 独立 load（因为 N 不同）。
- WGMMA **各发各的**，累加器各自独立。
- Epilogue 也各自独立做。

这相比 SM100 失去的：
- ❌ MMA issue 合并（SM100 只发一次 `2x1SM` MMA 指令，Hopper 两个 CTA 各发一次）。这个 cost 很小（issue rate 不是瓶颈）。
- ❌ M 维 UMMA_M = 256 的「自然放大」：SM100 一次 MMA 做 256×UMMA_N，Hopper 每 CTA 只能做 64×WGMMA_N（再在 warpgroup 内循环扩到 128）。不影响正确性，只影响 issue/wait overhead。

保留的：
- ✅ TMA multicast A，GMEM 带宽省 2×。
- ✅ B tile TMA overlap（两个 CTA 独立 load 权重）。
- ✅ Dispatch 可以跨 cluster 划分工作（token pool 读写）。

### 3.2 策略 B：取消 cluster，单 CTA

```
Cluster size 1:
- 每个 CTA 独立拉 A 和 B
- 取消 scheduler 的 2-block 约束
```

**优点**：代码简单，少一层同步（cluster barrier 不再需要）。
**缺点**：A tile TMA 带宽 2×。在 MoE 小 batch 场景里，A 本就是 FP8 少数据，多读一遍可以接受；但如果 B（权重）是瓶颈又被迫独立加载，就比较浪费。

**实用建议**：Hopper 版本 v0 可以先用策略 B（更简单），v1 再切到策略 A（补回 multicast 增益）。

## 4. Scheduler 需要改的地方

[scheduler/mega_moe.cuh](deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) 中有这样的约束：

```cpp
// 从 SM100 scheduler 概念性描述 (见 mega_moe_code_analysis.md §6.4)
static_assert(kNumL1BlockNs % 2 == 0);
static_assert(kNumL2BlockNs % 2 == 0);
static_assert(kNumSMs % 2 == 0);
```

原因是 cluster 中两个 CTA 要共享 M block、N 相邻。Hopper 的两种策略对应：

- **策略 A**：保留上述约束。
- **策略 B**：移除约束，scheduler 每次 `get_next_block` 只产出一个 block，block 连续递增。

## 5. Multicast TMA 的具体改动

[sm100_fp8_fp4_mega_moe.hpp](csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp) 中 L1 acts TMA descriptor 的创建：

```cpp
// SM100: multicast A to 2 CTAs within cluster
tensor_map_l1_acts_descriptor = make_tma_copy_desc_multicast<load_block_m, block_k, 2, true>(...);
```

Hopper 同样支持 multicast：
```cpp
// cp.async.bulk.tensor.3d.shared::cluster.global.multicast::cluster
```

TMA descriptor 构造函数层面 CUTLASS 已经封装好，改成 `num_multicast = cluster_size` 即可。但 Hopper multicast 要求：
- 所有接收 CTA 的 cluster 内 rank 必须由 `multicast_mask` 显式指定。
- TMA load 的 transaction barrier `arrive_and_expect_tx` 只在 leader CTA 设置字节数，其他 CTA 设 0。这个已经是 SM100 的做法，直接沿用。

## 6. Non-leader CTA 的 idle warp

SM100 上 [line 547-579](deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) 把 UMMA issue 放在 leader CTA 的 warp 6，非 leader CTA 的对应 warp 是 idle 的（等 TMEM barrier）。

Hopper 策略 A 下，两个 CTA 都要独立做 WGMMA，原本 non-leader CTA 的 idle warp 现在要干活。这意味着：
- warp 分配表对两个 CTA 是**完全对称**的（不再有 leader/non-leader 的行为差异）。
- Dispatch pull 的工作量在两个 CTA 之间平分（这本来就是对称的，没问题）。

## 7. Cluster 初始化序列

SM100 kernel 初始化阶段：
```cpp
cluster_sync_with_relaxed_arrive();  // 第 1 次：TMA descriptor prefetch 完成
// ... barrier init + TMEM allocation ...
cluster_sync_with_relaxed_arrive();  // 第 2 次：barrier + TMEM 状态同步
```

Hopper 上：
- 第 1 次 cluster sync 保留（prefetch TMA descriptor）。
- 第 2 次 cluster sync：**不再需要 TMEM allocation 同步**（没有 TMEM）；但 mbarrier init 仍然需要 cluster 级一致性。仍保留。

如果走策略 B（单 CTA），两次 `cluster_sync` 都可以换成 `__syncthreads()`。

## 8. 小结

- Hopper 无 2-CTA MMA，但有 cluster 和 multicast：cluster 只用来共享 A tile、节省 TMA 带宽。
- 推荐先落地 **策略 B（单 CTA）** 把 MMA/epilogue 跑通，再升级到 **策略 A（cluster + multicast）**。
- Scheduler 的 `kNumL1BlockNs/L2BlockNs/kNumSMs` 偶数约束依策略保留或移除。
- Leader/non-leader CTA 的分工消失，两个 CTA 对称。
- 初始化 cluster sync 保留，但去掉 TMEM 相关分支。
