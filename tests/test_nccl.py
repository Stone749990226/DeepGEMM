"""Quick NCCL sanity check. Usage:
    python tests/test_nccl.py [--nproc N]
"""
import argparse
import os
import socket
import sys
import time
import torch
import torch.distributed as dist


def worker(local_rank: int, nproc: int, port: int):
    print(f'[rank{local_rank}] pid={os.getpid()} gpu=cuda:{local_rank}', flush=True)

    torch.cuda.set_device(local_rank)

    # ---- init process group ------------------------------------------------
    t0 = time.time()
    print(f'[rank{local_rank}] calling init_process_group port={port}...', flush=True)
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        world_size=nproc,
        rank=local_rank,
    )
    print(f'[rank{local_rank}] init_process_group done in {time.time()-t0:.2f}s', flush=True)

    # ---- device check -------------------------------------------------------
    dev = torch.cuda.get_device_name(local_rank)
    print(f'[rank{local_rank}] device: {dev}', flush=True)

    # ---- all_reduce ---------------------------------------------------------
    x = torch.ones(4, device='cuda') * (local_rank + 1)
    dist.all_reduce(x)
    expected = sum(range(1, nproc + 1))
    assert x[0].item() == expected, f'all_reduce wrong: got {x[0].item()}, expected {expected}'
    print(f'[rank{local_rank}] all_reduce OK  (sum={x[0].item()})', flush=True)

    # ---- broadcast ----------------------------------------------------------
    y = torch.zeros(4, device='cuda')
    if local_rank == 0:
        y.fill_(42.0)
    dist.broadcast(y, src=0)
    assert y[0].item() == 42.0, f'broadcast wrong: got {y[0].item()}'
    print(f'[rank{local_rank}] broadcast OK', flush=True)

    # ---- P2P bandwidth (rank 0 <-> rank 1 only) -----------------------------
    dist.barrier()   # sync all ranks before P2P timing
    if nproc >= 2 and local_rank in (0, 1):
        MB = 256
        buf = torch.zeros(MB * 1024 * 1024 // 4, dtype=torch.float32, device='cuda')
        t1 = time.time()
        ITERS = 10
        for _ in range(ITERS):
            if local_rank == 0:
                dist.send(buf, dst=1)
                dist.recv(buf, src=1)
            else:
                dist.recv(buf, src=0)
                dist.send(buf, dst=0)
        elapsed = time.time() - t1
        bw_gb = (2 * ITERS * MB * 1e-3) / elapsed   # GB/s bidirectional
        print(f'[rank{local_rank}] P2P bandwidth rank0<->rank1: {bw_gb:.1f} GB/s '
              f'({MB} MB x {ITERS} iters)', flush=True)

    dist.barrier()
    if local_rank == 0:
        print('\nAll checks passed.', flush=True)

    dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--port', type=int, default=0,
                        help='TCP port for rendezvous (0 = auto-pick free port)')
    args = parser.parse_args()

    port = args.port if args.port else _find_free_port()
    print(f'Spawning {args.nproc} rank(s) on port {port}', flush=True)
    torch.multiprocessing.spawn(worker, args=(args.nproc, port), nprocs=args.nproc)
