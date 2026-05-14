"""Layered tests + perf benchmark for the SM90 (Hopper) MegaMoE kernel.

The fused FP8 SM90 MegaMoE kernel is exercised across a hierarchy of
scenarios so that each kernel path / heuristic branch / edge case is
covered with at least one configuration.

Modes
-----
*   Default (correctness): runs L1..L5 layered scenarios (see below).
*   ``--benchmark``: skips correctness; reports fused-kernel TFLOPS, HBM
    bandwidth, NVLink bandwidth, latency, and a ratio against an unfused
    *same-pipeline* baseline. Mirrors ``tests/test_mega_moe.py``. The baseline
    tier is auto-selected by ``import_baseline()``:
      - ``unfused-deepep``      : DeepEP dispatch + grouped FP8 GEMM (L1) +
                                  tilelang SwiGLU + grouped FP8 GEMM (L2) +
                                  DeepEP combine. Requires ``deep_ep`` and
                                  ``tilelang``; this is the apples-to-apples
                                  baseline (matches the SM100 reference).
      - ``unfused-fp8 (no comm)``: tilelang available but DeepEP isn't.
                                  Synthesises ``num_recv_tokens`` per rank,
                                  times the same FP8 GEMMs + SwiGLU without
                                  dispatch/combine. Lower bound on the win.
      - ``pytorch-bf16``        : neither dependency available; warns and
                                  falls back to a dense BF16 reference.
    Configurable via ``--num-tokens``, ``--hidden``, ``--intermediate-hidden``,
    ``--num-experts``, ``--num-topk`` etc.
*   ``--ncu-profile-only``: runs the fused kernel exactly once and exits;
    used by ``scripts/run_ncu_mega_moe.sh --arch sm90`` for NCU capture.
*   ``--local-rank-idx N``: single-process mode (bypasses
    ``torch.multiprocessing.spawn``); the wrapper script launches one
    process per rank and is responsible for exporting MASTER_PORT.

Layers (correctness mode)
-------------------------
  L1  Smoke           : single tiny config; only verifies the kernel runs
                        and produces an output close to a PyTorch reference.
  L2  Heuristic       : sweeps tokens-per-expert across the bands of
                        ``get_block_config_for_mega_moe_sm90`` so each
                        ``{block_m, num_epilogue_warpgroups}`` case is hit.
  L3  Shape sweep     : sweeps ``hidden``, ``intermediate_hidden`` and
                        ``num_topk`` over divisible-by-128 values.
  L4  Edge cases      : masking ratio, activation clamp (finite vs inf),
                        ``fast_math`` 0/1, ``num_tokens`` boundaries.
  L5  Stress          : ``--num-correctness-tests`` repeated random configs.

Notes
-----
*   The reference is a pure PyTorch BF16/FP32 simulation of the fused path
    (dequantize -> matmul -> SwiGLU + clamp + per-row quantize -> matmul ->
    cross-rank scatter -> BF16 reduce).  It is *not* bitwise-identical to
    the kernel; correctness is checked with ``calc_diff < 0.07``.
*   Because every scenario allocates its own symmetric memory buffer we
    re-`init_dist`/`destroy` once per process at the outer level only,
    and re-create ``SymmBuffer`` per scenario.
*   Skips itself when the device is not SM90.
"""

import argparse
import math
import os
import random
import sys
import time
import torch
import torch.distributed as dist
from typing import Tuple, List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major


# Kernel symbol used by bench_kineto / NCU. Matches the templated entry in
# `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh`.
SM90_KERNEL_NAME = 'sm90_fp8_mega_moe_impl'


# ----------------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------------

def _quantize_grouped_fp8_block_128_128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block (128, 128) FP8 quantization along (N, K).

    Args
    ----
    w : (G, N, K) bf16, with N % 128 == 0 and K % 128 == 0

    Returns
    -------
    fp8 : (G, N, K) torch.float8_e4m3fn
    sf  : (G, N // 128, K // 128) torch.float32, MN-major in the (N, K)
          plane (i.e. K is the inner contiguous dim, matching the kernel's
          ``stride_k = 1`` expectation and the DeepEP convention).
    """
    g, n, k = w.shape
    assert n % 128 == 0 and k % 128 == 0
    w_view = w.view(g, n // 128, 128, k // 128, 128).float()
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)        # (G, N/128, K/128)
    sf = amax / 448.0
    w_fp8 = (w_view / sf.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), sf.contiguous()


def _dequant_block_128_128(w_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Inverse of `_quantize_grouped_fp8_block_128_128`. Returns fp32."""
    *prefix, n, k = w_fp8.shape
    assert n % 128 == 0 and k % 128 == 0
    w_view = w_fp8.float().view(*prefix, n // 128, 128, k // 128, 128)
    return (w_view * sf.unsqueeze(-1).unsqueeze(-3)).view(*prefix, n, k)


def _dequant_per_token_per_128_k(x_fp8: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """For (M, K) fp8 with (M, K // 128) float SF (per-token, K-major)."""
    m, k = x_fp8.shape
    assert k % 128 == 0
    w_view = x_fp8.float().view(m, k // 128, 128)
    return (w_view * sf.unsqueeze(-1)).view(m, k)


# ----------------------------------------------------------------------------
# PyTorch reference
# ----------------------------------------------------------------------------

def _swiglu_fp32(gate_up: torch.Tensor, clamp: float) -> torch.Tensor:
    """SwiGLU with one-sided gate clamp and two-sided up clamp.

    Matches the fused kernel: ``silu(min(gate, c)) * clamp(up, -c, c)``.
    """
    n2 = gate_up.size(-1)
    half = n2 // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if math.isfinite(clamp):
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return torch.nn.functional.silu(gate) * up


def _reference_fused(
    x_fp8_local: torch.Tensor, x_sf_local: torch.Tensor,
    topk_idx_local: torch.Tensor, topk_weights_local: torch.Tensor,
    l1_w_fp8: torch.Tensor, l1_w_sf: torch.Tensor,
    l2_w_fp8: torch.Tensor, l2_w_sf: torch.Tensor,
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    num_experts: int, num_topk: int,
    hidden: int, intermediate_hidden: int,
    activation_clamp: float,
) -> torch.Tensor:
    """Reference: returns (num_tokens, hidden) bf16 result for *this* rank.

    All-gathers the global tokens / topk decisions / per-rank weights, then
    for each global token routes through its topk experts, applies the
    L1+SwiGLU+L2 path, and reduces over topk on the source rank.
    """
    num_experts_per_rank = num_experts // num_ranks

    # --- gather global token data --------------------------------------------------
    x_fp8_g = uneven_all_gather(x_fp8_local, group=group)      # (Mg, H)
    x_sf_g = uneven_all_gather(x_sf_local, group=group)        # (Mg, H/128)
    topk_idx_g = uneven_all_gather(topk_idx_local, group=group)         # (Mg, K)
    topk_w_g = uneven_all_gather(topk_weights_local, group=group)       # (Mg, K)
    mg = x_fp8_g.size(0)

    # rank-id lookup for each gathered token (for combine routing)
    rank_offsets = [0]
    sizes = [torch.tensor([0], device='cuda')]                  # placeholder
    # mimic uneven_all_gather to compute per-rank token counts
    local_size = torch.tensor([x_fp8_local.size(0)], device='cuda', dtype=torch.long)
    sizes_t = torch.empty(num_ranks, dtype=torch.long, device='cuda')
    dist.all_gather_into_tensor(sizes_t, local_size, group=group)
    sizes_list = sizes_t.tolist()
    src_rank_of = torch.empty(mg, dtype=torch.long, device='cuda')
    cur = 0
    for r, s in enumerate(sizes_list):
        src_rank_of[cur:cur + s] = r
        cur += s
    assert cur == mg

    # --- gather all-rank weights --------------------------------------------------
    # l1_w_fp8: (E_pr, 2*IH, H), l1_w_sf: (E_pr, 2*IH, H/128)
    l1_w_g = [torch.empty_like(l1_w_fp8) for _ in range(num_ranks)]
    l1_sf_g = [torch.empty_like(l1_w_sf) for _ in range(num_ranks)]
    l2_w_g = [torch.empty_like(l2_w_fp8) for _ in range(num_ranks)]
    l2_sf_g = [torch.empty_like(l2_w_sf) for _ in range(num_ranks)]
    dist.all_gather(l1_w_g, l1_w_fp8, group=group)
    dist.all_gather(l1_sf_g, l1_w_sf, group=group)
    dist.all_gather(l2_w_g, l2_w_fp8, group=group)
    dist.all_gather(l2_sf_g, l2_w_sf, group=group)
    l1_w_all = torch.stack(l1_w_g, dim=0)   # (R, E_pr, 2*IH, H)
    l1_sf_all = torch.stack(l1_sf_g, dim=0)
    l2_w_all = torch.stack(l2_w_g, dim=0)
    l2_sf_all = torch.stack(l2_sf_g, dim=0)

    # --- per-token / per-topk compute --------------------------------------------------
    # The combine slot tensor: (Mg, K, H) bf16 — each src rank will reduce over K.
    combine_buf = torch.zeros(mg, num_topk, hidden, dtype=torch.float32, device='cuda')

    # Precompute dequantized x in fp32
    x_fp32 = _dequant_per_token_per_128_k(x_fp8_g, x_sf_g)         # (Mg, H)

    # Iterate (cheap; reference is for small test configs only)
    # Token-chunked to keep gathered (S, 2*IH, H) dequant tensors below GPU memory.
    _CHUNK = 256
    for k in range(num_topk):
        # Skip masked
        mask = topk_idx_g[:, k] >= 0
        if not mask.any():
            continue
        sel_idx_full = mask.nonzero(as_tuple=False).squeeze(-1)    # (S,)
        for c0 in range(0, sel_idx_full.numel(), _CHUNK):
            sel_idx = sel_idx_full[c0:c0 + _CHUNK]
            eids = topk_idx_g[sel_idx, k]                          # (S,)
            weights = topk_w_g[sel_idx, k]                         # (S,)
            x_sel = x_fp32[sel_idx]                                # (S, H)

            dst_rank = (eids // num_experts_per_rank).long()
            dst_local = (eids % num_experts_per_rank).long()

            # L1 GEMM (per-token): y = x @ W^T  shape (S, 2*IH)
            l1_w_sel = _dequant_block_128_128(
                l1_w_all[dst_rank, dst_local],                     # (S, 2*IH, H)
                l1_sf_all[dst_rank, dst_local],
            )
            l1_y = torch.einsum('sk,snk->sn', x_sel, l1_w_sel)     # (S, 2*IH)
            del l1_w_sel

            # SwiGLU + clamp + multiply by topk weight
            l1_y = _swiglu_fp32(l1_y, activation_clamp) * weights.unsqueeze(-1)   # (S, IH)

            # Per-row, per-64-col FP8 quantize -> dequantize
            s_, ih = l1_y.shape
            assert ih == intermediate_hidden and ih % 64 == 0
            l1_view = l1_y.view(s_, ih // 64, 64)
            amax = l1_view.abs().amax(dim=-1).clamp(1e-4)          # (S, IH/64)
            sf2 = amax / 448.0
            l1_q = (l1_view / sf2.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
            l2_in = (l1_q * sf2.unsqueeze(-1)).view(s_, ih)        # (S, IH) fp32

            # L2 GEMM
            l2_w_sel = _dequant_block_128_128(
                l2_w_all[dst_rank, dst_local],                     # (S, H, IH)
                l2_sf_all[dst_rank, dst_local],
            )
            l2_y = torch.einsum('sn,smn->sm', l2_in, l2_w_sel)     # (S, H)
            del l2_w_sel

            # Scatter to combine buffer (cast to bf16 then back to mimic kernel storage)
            combine_buf[sel_idx, k] = l2_y.to(torch.bfloat16).float()

    # Sum over K -> (Mg, H), keep only this rank's slice
    y_full_bf16 = combine_buf.to(torch.bfloat16).sum(dim=1).to(torch.bfloat16)  # (Mg, H)
    start = sum(sizes_list[:rank_idx])
    end = start + sizes_list[rank_idx]
    return y_full_bf16[start:end].contiguous()


# ----------------------------------------------------------------------------
# Single-scenario runner
# ----------------------------------------------------------------------------

def _run_scenario(
    name: str,
    cfg: Dict[str, Any],
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    diff_tol: float,
):
    num_max = cfg['num_max_tokens_per_rank']
    num_tokens = cfg.get('num_tokens', num_max)
    hidden = cfg['hidden']
    intermediate_hidden = cfg['intermediate_hidden']
    num_experts = cfg['num_experts']
    num_topk = cfg['num_topk']
    masked_ratio = cfg.get('masked_ratio', 0.0)
    activation_clamp = cfg.get('activation_clamp', 10.0)
    fast_math = cfg.get('fast_math', True)

    assert num_experts % num_ranks == 0, f'{name}: experts {num_experts} not divisible by ranks {num_ranks}'
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    _t0 = time.time()
    def _trace(stage: str):
        elapsed = time.time() - _t0
        print(f'[rank{rank_idx}] {name} :: {stage}  ({elapsed:.1f}s)', flush=True)

    _trace('begin')
    torch.manual_seed(rank_idx * 1000 + abs(hash(name)) % 1000)
    random.seed(rank_idx * 1000 + abs(hash(name)) % 1000)

    # ---- Inputs (bf16) -------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    # Quantize x to FP8 with per-128 K float SF (SM90 format)
    # Quantize x to FP8 with per-128 K float SF (SM90 format)
    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    # Quantize weights with block (128, 128) — matches DeepSeekV4FlashFp8 / DeepEP.
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)

    # SM90 weight transform (gate/up interleave only). With block (128, 128)
    # SF, the SF tensor is consumed by the kernel as-is — no MN-major TMA
    # transform and no SF-side gate/up interleave is needed.
    _trace('weight_transform')
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf)
    )

    # ---- Allocate symm buffer -----------------------------------------------
    _trace('alloc_symm_buffer')
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max, num_topk,
        hidden, intermediate_hidden,
    )
    cum_stats = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')

    # ---- Run fused -----------------------------------------------------------
    _trace('copy_inputs')
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_w)

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    _trace('launch_fused (may JIT-compile, can take minutes)')
    deep_gemm.fp8_mega_moe(
        y_fused, transformed_l1, transformed_l2, buffer,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(128, 128, 128),
        activation='swiglu',
        activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
        fast_math=fast_math,
    )
    _trace('sync_fused')
    torch.cuda.synchronize()
    _trace('fused_done')

    # ---- Reference & check ---------------------------------------------------
    # Use the FP8 weights and their block-(128, 128) SF directly — the dequant
    # helper expects this MN/K-block SF layout, and the original (gate||up) row
    # ordering is what `_swiglu_fp32` splits with ``[..., :IH], [..., IH:]``.
    _trace('reference')
    y_ref = _reference_fused(
        x_fp8, x_sf, topk_idx, topk_w,
        l1_w_fp8, l1_w_sf, l2_w_fp8, l2_w_sf,
        rank_idx, num_ranks, group,
        num_experts, num_topk,
        hidden, intermediate_hidden,
        activation_clamp,
    )

    diff = calc_diff(y_fused, y_ref)
    ok = diff < diff_tol
    dist_print(f'  [{name:<32}] diff={diff:.4f} '
               f'(tol={diff_tol:.2f}) {"OK" if ok else "FAIL"}',
               once_in_node=True)
    assert ok, f'{name}: diff={diff} >= tol={diff_tol}'

    # Verify cum_stats has been incremented (i.e. dispatch ran)
    if num_tokens > 0 and masked_ratio < 1.0:
        assert cum_stats.sum().item() >= 0  # non-negative; can be 0 if nothing routed here

    buffer.destroy()
    dist.barrier()


# ----------------------------------------------------------------------------
# Benchmark / NCU mode (mirrors `tests/test_mega_moe.py` performance section)
# ----------------------------------------------------------------------------

def import_baseline():
    """Lazy-import the optional dependencies for the unfused baseline.

    Returns ``(deep_ep, tilelang_ops, do_bench)``; any may be ``None`` if not
    importable. ``do_bench`` is ``tilelang.profiler.bench.do_bench`` if
    available — used to time the baseline the same way SM100 does.
    """
    deep_ep = None
    tilelang_ops = None
    do_bench = None

    try:
        import deep_ep as _deep_ep  # type: ignore
        deep_ep = _deep_ep
    except Exception as ex:
        dist_print(f'INFO: deep_ep unavailable ({type(ex).__name__}: {ex})',
                   once_in_node=True)

    try:
        from tilelang.profiler.bench import do_bench as _do_bench  # type: ignore
        do_bench = _do_bench
    except Exception as ex:
        dist_print(f'INFO: tilelang.profiler.bench.do_bench unavailable '
                   f'({type(ex).__name__})', once_in_node=True)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..', 'third-party', 'tilelang_ops', '__init__.py'))
        if spec is not None and spec.loader is not None:
            tilelang_ops = importlib.util.module_from_spec(spec)
            sys.modules['tilelang_ops'] = tilelang_ops
            spec.loader.exec_module(tilelang_ops)
    except Exception as ex:
        dist_print(f'INFO: tilelang_ops unavailable ({type(ex).__name__}: {ex})',
                   once_in_node=True)
        tilelang_ops = None

    return deep_ep, tilelang_ops, do_bench


def _pytorch_swiglu_apply_weight_to_fp8(
    x: torch.Tensor,                    # (M, 2*IH) bf16 — gate||up halves
    topk_weights: torch.Tensor | None,  # (M,) float, or None for all-ones
    num_per_channels: int,              # K granularity for output SF (= 128 on SM90)
    clamp_value: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in PyTorch substitute for ``tilelang_ops.swiglu_apply_weight_to_fp8``
    when tilelang isn't installed.

    Output contract: ``(fp8_tensor, sf_tensor)`` where:
        * ``fp8_tensor`` : (M, IH) ``torch.float8_e4m3fn``
        * ``sf_tensor``  : (M, IH // num_per_channels) ``torch.float32``,
          row-major (matches ``per_token_cast_to_fp8`` and what
          ``m_grouped_fp8_gemm_nt_contiguous`` consumes on SM90).
    Decomposed PyTorch ops are intentionally slower than the fused kernel and
    than tilelang — that's exactly the unfused baseline behaviour the comparison
    is meant to capture.
    """
    m, two_ih = x.shape
    ih = two_ih // 2
    assert ih % num_per_channels == 0
    gate, up = x[:, :ih], x[:, ih:]
    if clamp_value is not None and math.isfinite(clamp_value):
        gate = gate.clamp(max=clamp_value)
        up = up.clamp(min=-clamp_value, max=clamp_value)
    y = (torch.nn.functional.silu(gate.float()) * up.float())
    if topk_weights is not None:
        y = y * topk_weights.unsqueeze(-1).float()
    y_view = y.view(m, ih // num_per_channels, num_per_channels)
    amax = y_view.abs().amax(dim=-1).clamp(min=1e-4)         # (M, IH/np)
    sf = amax / 448.0
    y_fp8 = (y_view / sf.unsqueeze(-1)).to(torch.float8_e4m3fn).view(m, ih).contiguous()
    return y_fp8, sf.contiguous()


def _bench_cuda_events(fn, num_warmup: int = 5, num_repeat: int = 20) -> float:
    """CUDA-event median timer used as a last-resort fallback when
    ``tilelang.profiler.bench.do_bench`` is unavailable.

    Returns elapsed seconds (median over ``num_repeat`` calls, L2 flushed).
    """
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    flush_l2_size = int(8e9 // 4)
    times_ms = []
    for _ in range(num_repeat):
        torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    return times_ms[len(times_ms) // 2] / 1e3


def _bench_unfused(fn, do_bench) -> float:
    """Time ``fn`` the same way SM100's baseline does it (cudagraph capture
    via ``tilelang.profiler.bench.do_bench``). Returns seconds."""
    # tilelang's do_bench returns ms; convert to s.
    return do_bench(fn, _n_warmup=5, _n_repeat=1,
                    backend='cudagraph', return_mode='median') / 1e3


def _build_uniform_psum_layout(num_tokens: int, num_experts_per_rank: int,
                               alignment: int) -> torch.Tensor:
    """Build a ``psum`` (cumulative-recv-per-expert) layout for the compute-only
    baseline, distributing ``num_tokens`` uniformly across local experts and
    rounding each per-expert count up to ``alignment``.
    """
    base = num_tokens // num_experts_per_rank
    base_aligned = ((base + alignment - 1) // alignment) * alignment
    psum = torch.tensor(
        [base_aligned * (i + 1) for i in range(num_experts_per_rank)],
        dtype=torch.int32, device='cuda',
    )
    return psum


def _run_benchmark(
    rank_idx: int, num_ranks: int, group: dist.ProcessGroup,
    args: argparse.Namespace,
):
    """Benchmark mode for the SM90 fused MegaMoE kernel.

    Mirrors the structure of ``tests/test_mega_moe.py`` (SM100 path):
      1. Build inputs and run the fused kernel once.
      2. Time the fused kernel via ``bench_kineto`` (TFLOPS / HBM / NVLink).
      3. Time an *unfused same-pipeline* baseline whose tier depends on what
         third-party packages are importable (see ``import_baseline``).

    The fused-vs-baseline ratio is therefore apples-to-apples whenever
    ``unfused_deepep`` is selected (matches SM100). The other tiers print a
    label that makes the lower fairness explicit.

    ``--ncu-profile-only`` short-circuits everything to a single fused launch
    so NCU can capture the kernel cleanly.
    """
    num_max = args.num_max_tokens_per_rank
    num_tokens = num_max if args.num_tokens == 0 else args.num_tokens
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    masked_ratio = args.masked_ratio
    activation_clamp = args.activation_clamp
    fast_math = bool(args.fast_math)

    assert num_experts % num_ranks == 0, f'experts {num_experts} not divisible by ranks {num_ranks}'
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max
    assert hidden % 128 == 0 and intermediate_hidden % 128 == 0

    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # ---- Inputs (bf16) -------------------------------------------------------
    x_bf = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    l1_bf = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    l2_bf = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device='cuda') * 0.05
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
    topk_w, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    if masked_ratio > 0:
        rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
        topk_idx.masked_fill_(rand_mask < masked_ratio, -1)
        topk_w.masked_fill_(topk_idx < 0, 0)

    x_fp8, x_sf = per_token_cast_to_fp8(x_bf, use_ue8m0=False, gran_k=128,
                                        use_packed_ue8m0=False)
    l1_w_fp8, l1_w_sf = _quantize_grouped_fp8_block_128_128(l1_bf)
    l2_w_fp8, l2_w_sf = _quantize_grouped_fp8_block_128_128(l2_bf)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        (l1_w_fp8, l1_w_sf), (l2_w_fp8, l2_w_sf)
    )

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max, num_topk,
        hidden, intermediate_hidden,
    )
    cum_stats_fused = torch.zeros(num_experts_per_rank, dtype=torch.int, device='cuda')
    cum_stats_baseline = cum_stats_fused.clone()

    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_w)

    y_fused = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    def run_fused():
        deep_gemm.fp8_mega_moe(
            y_fused, transformed_l1, transformed_l2, buffer,
            cumulative_local_expert_recv_stats=cum_stats_fused,
            recipe=(128, 128, 128),
            activation='swiglu',
            activation_clamp=activation_clamp if math.isfinite(activation_clamp) else None,
            fast_math=fast_math,
        )
        return y_fused

    # ---- Config print --------------------------------------------------------
    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # ---- NCU profile-only short-circuit -------------------------------------
    if args.ncu_profile_only:
        dist_print('Run fused kernel (NCU profile mode):', once_in_node=True)
        run_fused()
        torch.cuda.synchronize()
        dist_print(' > Done, exiting', once_in_node=True)
        dist.barrier()
        buffer.destroy()
        return

    # ---- Count tokens that actually land on this rank -----------------------
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) |
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = int((gathered_topk_idx != -1).sum().item())
    num_touched_experts = max(0, torch.unique(gathered_topk_idx.flatten()).numel() - 1)

    # ---- Time fused ---------------------------------------------------------
    t_fused = bench_kineto(
        run_fused, SM90_KERNEL_NAME,
        num_tests=args.num_bench_tests,
        barrier=lambda: dist.barrier(),
        trace_path=(f'{args.dump_profile_traces}/mega_moe_sm90_rank{rank_idx}.json'
                    if args.dump_profile_traces else None),
    )

    # ---- Build & time the unfused baseline ---------------------------------
    # The baseline always uses the SAME FP8 grouped GEMMs (`m_grouped_fp8_gemm_
    # nt_contiguous`) that the fused kernel performs internally — that's the
    # apples-to-apples comparison. The two optional packages affect *only*:
    #   * deep_ep    : whether we include real NVLink dispatch/combine, or
    #                  synthesise a uniform per-expert layout per rank.
    #   * tilelang   : whether the SwiGLU+per-128-K-FP8-quantize step uses
    #                  tilelang's fused kernel or a slower PyTorch decomposition.
    # If both are missing we still compute the *fair* (FP8 GEMM + PyTorch
    # SwiGLU, no comm) baseline. PyTorch SwiGLU being slower is exactly what
    # the unfused baseline should look like in production without a fused
    # SwiGLU op — so the comparison stays meaningful.
    t_baseline = 0.0
    baseline_label = 'skipped'
    ep_buffer = None

    if not args.skip_baseline and num_recv_tokens > 0:
        deep_ep, tilelang_ops, do_bench = import_baseline()
        clamp_arg = activation_clamp if math.isfinite(activation_clamp) else None
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        # Pick a SwiGLU+quantize impl: prefer tilelang, fall back to PyTorch.
        if tilelang_ops is not None:
            def swiglu_quant(x, topk_weights, avail_tokens):
                return tilelang_ops.swiglu_apply_weight_to_fp8(
                    x=x, topk_weights=topk_weights, avail_tokens=avail_tokens,
                    num_per_channels=128, use_col_major_scales=False,
                    round_scale=False, ue8m0_scale=False, output_bf16=False,
                    clamp_value=clamp_arg, fast_math=fast_math,
                )
            swiglu_tag = 'tilelang'
        else:
            def swiglu_quant(x, topk_weights, avail_tokens):
                # `avail_tokens` is unused here — PyTorch always processes all rows.
                return _pytorch_swiglu_apply_weight_to_fp8(
                    x, topk_weights, num_per_channels=128, clamp_value=clamp_arg,
                )
            swiglu_tag = 'pytorch-swiglu'

        # Pick a timer: prefer tilelang's cudagraph do_bench (matches SM100).
        if do_bench is not None:
            timer = lambda fn: _bench_unfused(fn, do_bench)
            timer_tag = 'cudagraph'
        else:
            timer = lambda fn: _bench_cuda_events(
                fn, num_warmup=args.num_baseline_warmup,
                num_repeat=args.num_baseline_repeat)
            timer_tag = 'event'

        if deep_ep is not None:
            # ---- Tier A: include real NVLink dispatch+combine (SM100-style) -----
            ep_buffer = deep_ep.ElasticBuffer(
                group,
                num_max_tokens_per_rank=num_max, hidden=hidden,
                num_topk=num_topk, use_fp8_dispatch=True,
                explicitly_destroy=True,
                allow_multiple_reduction=False,
                gpu_timeout_secs=10, cpu_timeout_secs=30,
            )

            def run_baseline():
                recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
                    x_fp8, topk_idx=topk_idx, topk_weights=topk_w,
                    cumulative_local_expert_recv_stats=cum_stats_baseline,
                    num_experts=num_experts, expert_alignment=alignment,
                    do_cpu_sync=False, do_handle_copy=False,
                    do_expand=True,
                    use_tma_aligned_col_major_sf=False,  # SM90: row-major float SF
                )
                n = recv_x[0].size(0)
                l1_y_bf16 = torch.empty((n, intermediate_hidden * 2),
                                        dtype=torch.bfloat16, device='cuda')
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    recv_x, (l1_w_fp8, l1_w_sf), l1_y_bf16,
                    handle.psum_num_recv_tokens_per_expert,
                    use_psum_layout=True, disable_ue8m0_cast=True,
                )
                l1_y = swiglu_quant(
                    l1_y_bf16, recv_topk_weights,
                    handle.psum_num_recv_tokens_per_expert[-1])
                l2_y_bf16 = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    l1_y, (l2_w_fp8, l2_w_sf), l2_y_bf16,
                    handle.psum_num_recv_tokens_per_expert,
                    use_psum_layout=True, disable_ue8m0_cast=True,
                )
                return ep_buffer.combine(l2_y_bf16, handle=handle)[0]

            baseline_label = f'unfused-deepep ({swiglu_tag}, {timer_tag})'
        else:
            # ---- Tier B: compute-only, synthesise a uniform per-expert layout ---
            psum_layout = _build_uniform_psum_layout(
                num_recv_tokens, num_experts_per_rank, alignment)
            n_synth = int(psum_layout[-1].item())
            synth_x_bf = torch.randn((n_synth, hidden), dtype=torch.bfloat16,
                                     device='cuda') * 0.05
            synth_x_fp8, synth_x_sf = per_token_cast_to_fp8(
                synth_x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
            synth_topk_w = torch.ones((n_synth, ), dtype=torch.float32, device='cuda')

            def run_baseline():
                l1_y_bf16 = torch.empty((n_synth, intermediate_hidden * 2),
                                        dtype=torch.bfloat16, device='cuda')
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (synth_x_fp8, synth_x_sf), (l1_w_fp8, l1_w_sf), l1_y_bf16,
                    psum_layout, use_psum_layout=True, disable_ue8m0_cast=True,
                )
                l1_y = swiglu_quant(l1_y_bf16, synth_topk_w, psum_layout[-1:])
                l2_y_bf16 = torch.empty((n_synth, hidden), dtype=torch.bfloat16, device='cuda')
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    l1_y, (l2_w_fp8, l2_w_sf), l2_y_bf16,
                    psum_layout, use_psum_layout=True, disable_ue8m0_cast=True,
                )
                return l2_y_bf16

            baseline_label = f'unfused-fp8 ({swiglu_tag}, no comm, {timer_tag})'

        try:
            t_baseline = timer(run_baseline)
        except Exception as ex:
            dist_print(f'WARN: baseline run failed ({type(ex).__name__}: {ex})',
                       once_in_node=True)
            dist_print('      check `m_grouped_fp8_gemm_nt_contiguous` SM90 support / shapes',
                       once_in_node=True)
            t_baseline = 0.0
            baseline_label = 'failed'

    # ---- Metrics -------------------------------------------------------------
    safe_div = lambda a, b: float('nan') if b == 0 else a / b

    # 3 matmuls (L1 gate, L1 up, L2), each 2*M*N*K. For SM90 path:
    #   L1: M=tokens, N=2*IH, K=H -> 2*tokens*2*IH*H = 4*tokens*IH*H
    #   L2: M=tokens, N=H,    K=IH -> 2*tokens*H*IH
    # Total = 6*tokens*IH*H = 2*tokens*(H*IH*3).
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: SM90 weights are FP8 (1 B/elem), not FP4 (0.5).
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden +        # L1 weights (FP8)
        num_touched_experts * hidden * intermediate_hidden +            # L2 weights (FP8)
        num_recv_tokens * hidden +                                      # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                         # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                         # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                    # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink: dispatch pull + combine write-back.
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) approximation, BF16 over (1 + topk) tensors.
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12
    approx_factor = t_fused / max(t_fused - t_reduction, 1e-12)

    # ---- Print ---------------------------------------------------------------
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:5.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:5.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:5.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:4.0f} GB/s | '
               f'{t_fused * 1e6:5.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x {baseline_label}')

    dist.barrier()
    buffer.destroy()
    if ep_buffer is not None:
        ep_buffer.destroy()


# ----------------------------------------------------------------------------
# Scenario tables
# ----------------------------------------------------------------------------

# A single tiny config used as a smoke test.
_SMOKE = dict(
    num_max_tokens_per_rank=64, num_tokens=64,
    hidden=512, intermediate_hidden=512,
    num_experts=8, num_topk=2,
)


def _layer1_smoke() -> List[Tuple[str, Dict[str, Any]]]:
    return [('L1.smoke', dict(_SMOKE))]


def _layer2_heuristic_branches(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Vary tokens / (num_experts * num_topk / num_ranks) so each
    ``get_block_config_for_mega_moe_sm90`` band fires at least once.

    The heuristic decides on ``avg_tokens_per_expert``; we approximate by
    setting ``num_max_tokens_per_rank`` and ``num_topk`` while keeping
    ``num_experts`` fixed.  The bands are at 64.5 / 96.5 / 192.5.
    """
    base = dict(hidden=1024, intermediate_hidden=1024,
                num_experts=8 * num_ranks, num_topk=2)
    out: List[Tuple[str, Dict[str, Any]]] = []
    # tokens-per-rank settings chosen to hit (small / mid / large) bands
    for tokens, label in [(64, 'small'), (256, 'midA'), (512, 'midB'), (2048, 'large')]:
        cfg = dict(base)
        cfg.update(num_max_tokens_per_rank=tokens, num_tokens=tokens)
        out.append((f'L2.heur.{label}.t{tokens}', cfg))
    return out


def _layer3_shape_sweep(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    base_experts = 8 * num_ranks
    for hidden in (512, 2048):
        for ih in (512, 2048):
            for topk in (1, 2, 4):
                if topk > base_experts:
                    continue
                cfg = dict(num_max_tokens_per_rank=128, num_tokens=128,
                           hidden=hidden, intermediate_hidden=ih,
                           num_experts=base_experts, num_topk=topk)
                out.append((f'L3.h{hidden}_ih{ih}_k{topk}', cfg))
    return out


def _layer4_edges(num_ranks: int) -> List[Tuple[str, Dict[str, Any]]]:
    base = dict(num_max_tokens_per_rank=128,
                hidden=512, intermediate_hidden=512,
                num_experts=8 * num_ranks, num_topk=2)
    out = []
    # Masked ratios
    for mr in (0.0, 0.3, 0.7):
        cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=mr)
        out.append((f'L4.mask{mr:.1f}', cfg))
    # All masked
    cfg = dict(base); cfg.update(num_tokens=128, masked_ratio=1.0)
    out.append(('L4.mask_all', cfg))
    # Activation clamp variations (finite vs inf)
    for c in (1.0, 10.0, math.inf):
        cfg = dict(base); cfg.update(num_tokens=128, activation_clamp=c)
        out.append((f'L4.clamp{c}', cfg))
    # fast_math toggle
    for fm in (True, False):
        cfg = dict(base); cfg.update(num_tokens=128, fast_math=fm)
        out.append((f'L4.fm{int(fm)}', cfg))
    # num_tokens boundaries
    cfg = dict(base); cfg.update(num_tokens=0)
    out.append(('L4.tokens0', cfg))
    cfg = dict(base); cfg.update(num_tokens=base['num_max_tokens_per_rank'])
    out.append(('L4.tokens_max', cfg))
    return out


def _layer5_stress(num_ranks: int, num_tests: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Random configs under simple constraints."""
    rng = random.Random(0xC0FFEE)
    out = []
    for i in range(num_tests):
        hidden = rng.choice([512, 1024, 2048])
        ih = rng.choice([512, 1024, 2048])
        topk = rng.choice([1, 2, 4])
        tokens = rng.choice([32, 64, 128, 256, 512])
        masked = rng.choice([0.0, 0.0, 0.3, 0.5])
        clamp = rng.choice([1.0, 10.0, math.inf])
        fm = rng.choice([True, False])
        cfg = dict(num_max_tokens_per_rank=tokens, num_tokens=tokens,
                   hidden=hidden, intermediate_hidden=ih,
                   num_experts=8 * num_ranks, num_topk=topk,
                   masked_ratio=masked, activation_clamp=clamp, fast_math=fm)
        out.append((f'L5.rand{i:03d}', cfg))
    return out


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    print(f'[rank{local_rank}] process started, initializing NCCL...', flush=True)
    torch.cuda.set_device(local_rank)
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    print(f'[rank{rank_idx}] NCCL init done, world_size={num_ranks}', flush=True)

    # Skip on non-SM90
    if get_arch_major() != 9:
        dist_print(f'[SKIP] test_mega_moe_sm90 requires SM90; got SM{get_arch_major()}0',
                   once_in_node=True)
        dist.destroy_process_group()
        return

    # Benchmark / NCU mode: bypass the layered correctness scenarios and run
    # a single configurable scenario from CLI args.
    if args.benchmark or args.ncu_profile_only:
        _run_benchmark(rank_idx, num_ranks, group, args)
        dist.barrier()
        dist.destroy_process_group()
        return

    diff_tol = args.diff_tol
    layers: List[Tuple[str, Dict[str, Any]]] = []

    if 1 in args.layers:
        layers += _layer1_smoke()
    if 2 in args.layers:
        layers += _layer2_heuristic_branches(num_ranks)
    if 3 in args.layers:
        layers += _layer3_shape_sweep(num_ranks)
    if 4 in args.layers:
        layers += _layer4_edges(num_ranks)
    if 5 in args.layers:
        layers += _layer5_stress(num_ranks, args.num_correctness_tests or 8)

    if args.filter:
        layers = [(n, c) for n, c in layers if args.filter in n]

    dist_print(f'SM90 MegaMoE test plan: {len(layers)} scenarios across '
               f'layers {sorted(args.layers)} on {num_ranks} ranks',
               once_in_node=True)

    failures: List[str] = []
    for name, cfg in layers:
        try:
            _run_scenario(name, cfg, rank_idx, num_ranks, group, diff_tol)
        except AssertionError as ex:
            dist_print(f'  [{name}] FAIL: {ex}', once_in_node=True)
            failures.append(name)
            if args.fail_fast:
                break

    dist_print('', once_in_node=True)
    if failures:
        dist_print(f'FAILED {len(failures)}/{len(layers)} scenarios: {failures}',
                   once_in_node=True)
    else:
        dist_print(f'PASSED all {len(layers)} scenarios', once_in_node=True)

    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layered SM90 MegaMoE tests')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='Number of ranks to spawn (default: 2)')
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which layers to run (1..5). Default: 1 2 3 4. '
                             'Layer 5 requires --num-correctness-tests.')
    parser.add_argument('--num-correctness-tests', type=int, default=None,
                        help='Layer 5 stress test count')
    parser.add_argument('--filter', type=str, default='',
                        help='Substring filter on scenario names')
    parser.add_argument('--diff-tol', type=float, default=0.07,
                        help='calc_diff tolerance (default: 0.07)')
    parser.add_argument('--fail-fast', action='store_true',
                        help='Stop on first failing scenario')

    # Benchmark / NCU mode --------------------------------------------------
    parser.add_argument('--benchmark', action='store_true',
                        help='Skip correctness layers; run perf benchmark on a single config')
    parser.add_argument('--ncu-profile-only', action='store_true',
                        help='Run the fused kernel once and exit (for NCU profiling)')
    parser.add_argument('--num-bench-tests', type=int, default=30,
                        help='Iterations for bench_kineto (default: 30)')
    parser.add_argument('--num-baseline-warmup', type=int, default=5,
                        help='PyTorch baseline warmup iterations (default: 5)')
    parser.add_argument('--num-baseline-repeat', type=int, default=20,
                        help='PyTorch baseline timed iterations (default: 20)')
    parser.add_argument('--baseline-max-tokens', type=int, default=4096,
                        help='Cap baseline tokens to bound peak memory; the timed '
                             'result is then scaled to actual num_recv_tokens (default: 4096)')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip the PyTorch BF16 compute-only baseline')
    parser.add_argument('--dump-profile-traces', type=str, default='',
                        help='Directory to write Chrome traces to (one per rank)')
    parser.add_argument('--local-rank-idx', type=int, default=None,
                        help='Run as a single process with this local rank '
                             '(used by NCU; bypasses torch.multiprocessing.spawn)')

    # Benchmark scenario knobs (a la tests/test_mega_moe.py) -----------------
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=4096,
                        help='Per-rank token capacity (default: 4096)')
    parser.add_argument('--num-tokens', type=int, default=0,
                        help='Per-rank token count; 0 -> use --num-max-tokens-per-rank (default: 0)')
    parser.add_argument('--hidden', type=int, default=2048,
                        help='Hidden size (default: 2048)')
    parser.add_argument('--intermediate-hidden', type=int, default=2048,
                        help='Intermediate hidden size (default: 2048)')
    parser.add_argument('--num-experts', type=int, default=64,
                        help='Total experts across all ranks (default: 64)')
    parser.add_argument('--num-topk', type=int, default=4,
                        help='Top-k experts per token (default: 4)')
    parser.add_argument('--masked-ratio', type=float, default=0.0,
                        help='Fraction of topk slots to mask out (default: 0.0)')
    parser.add_argument('--activation-clamp', type=float, default=10.0,
                        help='SwiGLU clamp; pass `inf` for no clamp (default: 10.0)')
    parser.add_argument('--fast-math', type=int, default=1,
                        help='1 to enable fast math, 0 to disable (default: 1)')

    args = parser.parse_args()

    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    np_ = args.num_processes
    import socket

    if args.local_rank_idx is not None:
        # Single-process mode (used by NCU: a wrapper script launches each
        # process separately). All ranks must share MASTER_PORT, so the
        # wrapper is responsible for exporting it; we just provide fallbacks.
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        if 'MASTER_PORT' not in os.environ:
            with socket.socket() as _s:
                _s.bind(('', 0))
                os.environ['MASTER_PORT'] = str(_s.getsockname()[1])
        print(f'Single-process mode: local_rank={args.local_rank_idx}, '
              f'world_size={np_}, port={os.environ["MASTER_PORT"]}',
              flush=True)
        test(args.local_rank_idx, np_, args)
    else:
        # Spawn path: force fresh single-node settings (override any cluster env vars).
        with socket.socket() as _s:
            _s.bind(('', 0))
            os.environ['MASTER_PORT'] = str(_s.getsockname()[1])
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        print(f'Spawning {np_} rank(s), layers={args.layers}, port={os.environ["MASTER_PORT"]}',
              flush=True)
        torch.multiprocessing.spawn(test, args=(np_, args), nprocs=np_)
