# `tests/test_mega_moe.py` 测试逻辑梳理

本文对 [tests/test_mega_moe.py](../tests/test_mega_moe.py) 的整体流程做分段解析，便于理解 Mega MoE 的测试骨架、对照基线、正确性校验与性能度量。

---

## 1. 总体定位

该脚本验证 DeepGEMM 提供的 **融合版 Mega MoE**（`deep_gemm.fp8_fp4_mega_moe`）：把 "EP dispatch → 两个分组 GEMM (+SwiGLU) → EP combine" 整条链路融合为单一 kernel。测试从两个维度评估：

- **正确性**：与 legacy 非融合路径（`deep_ep` + `tilelang_ops` + 两个 `m_grouped_fp8_fp4_gemm_nt_contiguous`）逐 bit 对齐；
- **性能**：融合 kernel vs 旧路径的端到端时延、TFLOPS、HBM / NVLink 带宽。

入口 [test_mega_moe.py:285-295](../tests/test_mega_moe.py#L285-L295) 使用 `torch.multiprocessing.spawn` 拉起 `num_processes` 个进程；也支持 `--local-rank-idx` 单进程模式以便 `ncu` 做逐 rank 采样。

---

## 2. 基线加载：`import_baseline` ([L14-L32](../tests/test_mega_moe.py#L14-L32))

用 `importlib.util` 从 `third-party/tilelang_ops/__init__.py` 动态加载老实现，并尝试导入 `deep_ep` 和 `tilelang.profiler.bench.do_bench`。

- 任一依赖缺失 → `is_legacy_loaded = False`，后续只跑融合 kernel 并跳过对照。
- 返回 `(deep_ep, tilelang_ops, do_bench, is_legacy_loaded)` 四元组。

这样设计让该脚本在没有 legacy 环境的机器上仍可用于纯 profiling。

---

## 3. 进程初始化与配置 ([L36-L50](../tests/test_mega_moe.py#L36-L50))

- `init_dist` 建立 NCCL 进程组，拿到 `(rank_idx, num_ranks, group)`；
- 随机种子用 `rank_idx`，保证不同 rank 生成不同 token/路由但各自确定性；
- 关键超参：`num_max_tokens_per_rank`、`num_tokens`（0 时 = max - random remove）、`hidden`、`intermediate_hidden`、`num_experts`、`num_topk`；
- `num_experts_per_rank = num_experts // num_ranks`——每个 rank 承担均匀切分的一段专家。

---

## 4. 对称显存缓冲区 ([L52-L57](../tests/test_mega_moe.py#L52-L57))

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden)
```

一次性申请融合路径所需的所有 symmetric buffer（x、x_sf、topk_idx、topk_weights 等），`buffer.buffer.nbytes` 会在日志里打印成 GiB。融合 kernel 内部依赖这块共享显存完成 EP dispatch/combine。

---

## 5. 输入构造 `create_inputs` ([L59-L97](../tests/test_mega_moe.py#L59-L97))

通过 `global` 声明把生成的张量暴露给 `run_fused`/`run_baseline` 复用，含：

1. **Token 与权重**：`x (BF16)`、`l1_weights (E_local, 2H_i, H)`、`l2_weights (E_local, H, H_i)`；
2. **路由**：对随机 `scores` 做 topk 得到 `topk_idx/topk_weights`；按 `masked_ratio` 把部分 topk 置 -1 并把对应权重置 0，用于模拟丢弃 token；
3. **接收统计**：`cumulative_local_expert_recv_stats_*`——融合与基线各一份克隆，保证 in-place 累加结果可比对；
4. **数值类型转换**：
   - 激活：`per_token_cast_to_fp8(..., use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`；
   - 权重：逐 group 调用 `per_token_cast_to_fp4` 得到 FP4+UE8M0 SF，再用 `transform_sf_into_required_layout` 调成 MN-major；
5. **融合专用布局**：`deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)` 得到 `transformed_l1_weights / transformed_l2_weights`，供融合 kernel 直接消费。

前置断言：`hidden % 128 == 0`、`intermediate_hidden % 128 == 0`、权重 K 维 % 128 == 0（FP8/FP4 分块 scale 的硬性要求）。

---

## 6. 融合路径 `run_fused` ([L99-L116](../tests/test_mega_moe.py#L99-L116))

1. 把 `x / x_sf / topk_idx / topk_weights` **拷贝进 symmetric buffer**（注释指出 debug 模式会清零整块 buffer，所以每次都要重灌）；
2. 分配输出 `y (num_tokens, hidden) BF16`；
3. 单次调用 `deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buffer, cumulative_local_expert_recv_stats=..., activation_clamp, fast_math)` 完成 dispatch+两 GEMM+SwiGLU+combine；
4. 返回 `(y, cumulative_local_expert_recv_stats_fused)` 便于 bitwise 比对。

---

## 7. 基线路径 `run_baseline` ([L140-L171](../tests/test_mega_moe.py#L140-L171))

仅在 `is_legacy_loaded` 时构造，使用：

- `deep_ep.ElasticBuffer` 负责 EP `dispatch`/`combine`，设置 `use_fp8_dispatch=True`、`allow_multiple_reduction=False`、GPU/CPU 超时等；
- `deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)`：先查询理论对齐再强制设置，给 `m_grouped_fp8_fp4_gemm_nt_contiguous` 用；

流程顺序：

1. `ep_buffer.dispatch(..., do_expand=True, use_tma_aligned_col_major_sf=True)` 拿到 `recv_x`、`recv_topk_weights`、`handle`；
2. L1 GEMM：`m_grouped_fp8_fp4_gemm_nt_contiguous(recv_x, l1_weights, l1_y, psum_num_recv_tokens_per_expert, use_psum_layout=True, recipe=(1,1,32))`；
3. `tilelang_ops.swiglu_apply_weight_to_fp8`：SwiGLU 激活 + 应用 topk 权重，并把输出再量化回 FP8 列主序 SF；
4. L2 GEMM：同上 GEMM 接口；
5. `ep_buffer.combine(l2_y, handle=handle)[0]` 完成 all-to-all reduce；
6. 返回 `(combined_output, cumulative_local_expert_recv_stats_baseline)`。

这是业界常见的 "非融合 EP+GEMM+combine" 参考实现。

---

## 8. 正确性测试 ([L174-L188](../tests/test_mega_moe.py#L174-L188))

若 legacy 加载成功且 `num_correctness_tests > 0`：

- 循环 `create_inputs()`（每次刷新随机状态），分别跑 `run_fused` 与 `run_baseline`；
- 对二者返回的每个 tensor 使用 **`torch.equal`** 严格比对（要求完全 bitwise 相等，不是 `allclose`）；
- 每 100 次或末尾打印进度。

`--ncu-profile-only` 分支 ([L120-L130](../tests/test_mega_moe.py#L120-L130)) 完全跳过基线和正确性环节，只跑一次 fused kernel 供 Nsight Compute 采样。

---

## 9. 接收 token 统计 ([L191-L194](../tests/test_mega_moe.py#L191-L194))

用 `uneven_all_gather` 把每个 rank 的 `topk_idx` 聚集到所有 rank；将不属于本 rank 专家范围的条目置 -1；`num_recv_tokens = (!=-1).sum()` 即本 rank 实际要算的 token 路由数，后续做 FLOPS/带宽分母。

---

## 10. 基准测量 ([L197-L201](../tests/test_mega_moe.py#L197-L201))

- **Fused**：`deep_gemm.testing.bench_kineto(run_fused, 'mega_moe', barrier=..., trace_path=...)` 通过 Kineto 抓取命名为 `mega_moe` 的 kernel 时长；barrier 优先用 `ep_buffer.barrier`，否则退化为 `dist.barrier`；
- **Baseline**：`tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3` 得到秒级时延，未加载 legacy 时置 0。

---

## 11. 性能指标计算 ([L203-L228](../tests/test_mega_moe.py#L203-L228))

1. **TFLOPS**：`2 * num_recv_tokens * (H * H_i * 3) / 1e12 / t`，三次矩阵乘 L1-left/L1-right/L2（因 `intermediate_hidden*2`）合并记为 `H*H_i*3`；
2. **HBM bytes**：按精度逐项累加
   - L1/L2 权重（FP4 = 0.5B） × 触达的专家数；
   - L1 读/写（FP8=1B）、L2 读（FP8）、最终输出（BF16=2B） × 接收 token 数；
3. **NVLink bytes**：`num_recv_tokens * hidden * 3`——dispatch 拉取 + combine 回写估算；
4. **Reduction 时间近似**：`num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12`（以 6.5 TB/s 为单位带宽反算串行 reduce 时间）；
5. **Overlap 放大因子**：`approx_factor = t_fused / (t_fused - t_reduction)`，用于估算扣除 reduce 后的纯计算/通信并行上限，并对 TFLOPS/HBM/NVL 三项做同样放大展示。

`safe_div` 避免 benchmark 时长为 0 时除零。

---

## 12. 结果输出与清理 ([L230-L250](../tests/test_mega_moe.py#L230-L250))

每个 rank 打印一行：

```
EP: r/R | xxx TFLOPS | overlap: xxx TFLOPS, HBM xxx GB/s, NVL xxx GB/s | t us, reduction: t us | k.kx legacy
```

`k.kx legacy` 是 `t_baseline / t_fused` 的倍率（>1 表示融合更快）。

最后 `dist.barrier()` → `buffer.destroy()` → `ep_buffer.destroy()`（若有）→ `destroy_process_group()` 保证多进程干净退出。

---

## 13. CLI 参数 ([L253-L283](../tests/test_mega_moe.py#L253-L283))

分三组：

- **资源**：`--ncu-profile-only`、`--num-processes`；
- **模型**：token/hidden/expert/topk/masked-ratio/activation-clamp/fast-math；
- **测试**：`--num-correctness-tests`、`--dump-profile-traces`（自动 `os.makedirs`）、`--local-rank-idx`（给 NCU 单进程跑）。

---

## 14. 一张流程图（文字版）

```
for each rank:
  init_dist → symm_buffer (fp8_fp4_mega_moe)
  create_inputs()
      ├─ x (BF16 → FP8/UE8M0)
      ├─ l1/l2 weights (BF16 → FP4 per-group) → transform_weights_for_mega_moe
      └─ topk_idx/weights (+ mask)

  [if legacy]
    build ep_buffer (deep_ep)
    correctness loop:
        run_fused()      ── torch.equal ──▶
        run_baseline()   ──              ──▶ must bitwise match

  num_recv_tokens via uneven_all_gather
  t_fused    = bench_kineto(run_fused, 'mega_moe')
  t_baseline = tilelang do_bench(run_baseline)

  compute TFLOPS / HBM GB/s / NVL GB/s / reduction time
  print per-rank summary → barrier → destroy
```

---

## 15. 关键设计点 & 阅读提示

1. **Bitwise 对齐**：`torch.equal` 而非 `torch.allclose`——融合 kernel 必须与基线位级一致，暗示两条路径使用完全相同的量化与累加顺序。
2. **`global` 变量共享输入**：在函数内用 `global` 暴露输入张量，省去重复传参；但代价是测试不是可重入的，`create_inputs` 的每次调用会覆盖全局。
3. **`cumulative_local_expert_recv_stats` 的两份克隆**：这个张量是 **in-place 累加** 的，所以必须给两条路径各备份一份才能公平比对。
4. **symmetric buffer 必须每次重灌**：`run_fused` 开头把输入 copy 进 buffer——注释说明 debug 模式会在每次调用前清零 buffer。
5. **`approx_factor` 的含义**：这是一种"扣除掉串行 reduce 的理论上限"估算，用于展示融合 + overlap 理想情况下可达到的 TFLOPS/带宽，对实际业务性能解读要小心。
6. **分发与 GEMM 对齐**：baseline 需要先 `set_mk_alignment_for_contiguous_layout(get_theoretical_mk_alignment_for_contiguous_layout())`，这是 `m_grouped_fp8_fp4_gemm_nt_contiguous` 在 `use_psum_layout=True` 下的硬约束。
7. **SM90 支持**：文件顶部 TODO 指出"需要给 SM90 跳过测试"，当前脚本默认只保证 SM100（Blackwell）跑得动。
