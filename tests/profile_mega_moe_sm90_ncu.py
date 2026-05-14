"""
Nsight Compute (ncu) profiling script for deep_gemm.fp8_mega_moe on SM90 / Hopper.

Usage
-----
# 普通正确性/冒烟运行，脚本内部用 torch.multiprocessing.spawn 启动所有 rank：
CUDA_VISIBLE_DEVICES=0,2,3,4 \\
python tests/profile_mega_moe_sm90_ncu.py --num-processes 4

# NCU profiling 推荐每个 rank 外面包一个 ncu 进程，并使用 application replay + lockstep。
# 这能避免 ncu kernel replay 把带 NVLink barrier 的多 rank kernel 重放到不同步。
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export DG_JIT_WITH_LINEINFO=1
NUM_PROCESSES=4
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    CUDA_VISIBLE_DEVICES=0,2,3,4 \\
    ncu --config-file off \\
        --force-overwrite \\
        --kernel-name sm90_fp8_mega_moe_impl \\
        --profile-from-start off \\
        --replay-mode application \\
        --lockstep-kernel-launch \\
        --communicator tcp \\
        --communicator-tcp-num-peers ${NUM_PROCESSES} \\
        --set full \\
        --launch-skip 0 \\
        --launch-count 1 \\
        -o mega_moe_sm90.${i} \\
        python tests/profile_mega_moe_sm90_ncu.py \\
            --local-rank-idx ${i} \\
            --num-processes ${NUM_PROCESSES} &
done
wait

# 只看 roofline 相关 metrics（更快）时，把 --set full 换成：
#     --section SpeedOfLight --section MemoryWorkloadAnalysis

注意
----
- 不推荐对本脚本的 spawn 模式直接套 `ncu --target-processes all --set full`：
  ncu kernel replay 可能让各 rank 的 NVLink barrier 失配并触发 timeout。
- `--local-rank-idx` 用于外部 launcher：每个 rank 一个 Python 进程，所有进程必须共享同一个
  MASTER_ADDR/MASTER_PORT，且 ncu 的 --communicator-tcp-num-peers 要等于 --num-processes。
- --profile-from-start off 配合脚本内 cudaProfilerStart/Stop，
  只对 warmup 之后的单次 fp8_mega_moe 调用做 profiling，避免初始化噪声
"""

import argparse
import math
import random

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import init_dist

FP8_E4M3_MAX = 448.0


def _quantize_grouped_fp8_block_128_128(w: torch.Tensor):
    g, n, k = w.shape
    w_view = w.view(g, n // 128, 128, k // 128, 128).float()
    amax = w_view.abs().amax(dim=(-1, -3)).clamp(1e-4)
    sf = amax / FP8_E4M3_MAX
    w_fp8 = (w_view / sf.unsqueeze(-1).unsqueeze(-3)).to(torch.float8_e4m3fn)
    return w_fp8.view(g, n, k).contiguous(), sf.contiguous()


def profile(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    num_max_tokens = args.num_max_tokens_per_rank
    num_tokens = args.num_tokens if args.num_tokens > 0 else num_max_tokens
    hidden = args.hidden
    intermediate_hidden = args.intermediate_hidden
    num_experts = args.num_experts
    num_topk = args.num_topk
    num_experts_per_rank = num_experts // num_ranks

    assert hidden % 128 == 0
    assert intermediate_hidden % 128 == 0
    assert intermediate_hidden // 64 <= 64, (
        f"SM90 fused kernel 要求 intermediate_hidden <= 4096, 当前 {intermediate_hidden}"
    )

    # ---- 输入 & 权重 ----
    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_w_bf16 = torch.randn(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.bfloat16, device="cuda",
    )
    l2_w_bf16 = torch.randn(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.bfloat16, device="cuda",
    )

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)

    # ---- 量化 ----
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_w_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_w_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights
    )

    clamp_arg = args.activation_clamp if math.isfinite(args.activation_clamp) else None

    # ---- SymmBuffer & 输出 ----
    sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens, num_topk, hidden, intermediate_hidden,
    )
    cum_stats = torch.zeros((num_experts_per_rank,), dtype=torch.int, device="cuda")
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    def run_once():
        sym_buffer.x[:num_tokens].copy_(x_fp8[0])
        sym_buffer.x_sf[:num_tokens].copy_(x_fp8[1])
        sym_buffer.topk_idx[:num_tokens].copy_(topk_idx)
        sym_buffer.topk_weights[:num_tokens].copy_(topk_weights)
        deep_gemm.fp8_mega_moe(
            y,
            transformed_l1,
            transformed_l2,
            sym_buffer,
            cumulative_local_expert_recv_stats=cum_stats,
            recipe=(128, 128, 128),
            activation="swiglu",
            activation_clamp=clamp_arg,
            fast_math=bool(args.fast_math),
        )
        return y

    # ---- warmup（ncu profile-from-start off 期间运行，不会被 profiler 捕获）----
    for _ in range(args.num_warmup):
        run_once()
    dist.barrier()
    torch.cuda.synchronize()

    # ---- 打开 profiler 窗口，只 profile 一次 kernel 调用 ----
    torch.cuda.cudart().cudaProfilerStart()
    run_once()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    dist.barrier()
    sym_buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ncu profiling for fp8_mega_moe (SM90)")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=8192)
    parser.add_argument("--num-tokens", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate-hidden", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--num-warmup", type=int, default=3, help="profiler 窗口前的 warmup 次数")
    parser.add_argument(
        "--local-rank-idx",
        type=int,
        default=None,
        help="外部 launcher/ncu 启动单个 rank 时使用；未设置则由本脚本 spawn 所有 rank",
    )
    args = parser.parse_args()

    if args.local_rank_idx is None:
        torch.multiprocessing.spawn(
            profile, args=(args.num_processes, args), nprocs=args.num_processes
        )
    else:
        if not 0 <= args.local_rank_idx < args.num_processes:
            raise ValueError(
                f"--local-rank-idx must be in [0, {args.num_processes}), "
                f"got {args.local_rank_idx}"
            )
        profile(args.local_rank_idx, args.num_processes, args)
