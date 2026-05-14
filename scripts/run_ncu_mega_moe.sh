#!/bin/bash
#
# Run NCU profiling on the fused MegaMoE kernel.
#
# Usage:
#   scripts/run_ncu_mega_moe.sh --arch sm100 [--num-processes 8] [-o work] [python args...]
#   scripts/run_ncu_mega_moe.sh --arch sm90  [--num-processes 2] [-o work] [python args...]
#
# `--arch sm100` profiles the SM100 (Blackwell) FP8/FP4 kernel
# (`sm100_fp8_fp4_mega_moe_impl`) via `tests/test_mega_moe.py`.
#
# `--arch sm90`  profiles the SM90 (Hopper) FP8 kernel
# (`sm90_fp8_mega_moe_impl`) via `tests/test_mega_moe_sm90.py --benchmark`.
# A unique MASTER_PORT is exported so all NCU child processes share the same
# rendezvous (the test scripts no longer pick one per-process in this mode).

set -e

# Defaults
arch=""
num_processes=""        # default depends on --arch
output_dir=work
python_args=()

# Parse args; intercept --arch / --num-processes / -o, forward the rest.
for ((arg_idx = 1; arg_idx <= $#; ++arg_idx)); do
    arg="${!arg_idx}"
    case "$arg" in
        --arch)
            if ((arg_idx < $#)); then
                ((arg_idx++))
                arch="${!arg_idx}"
            fi
            ;;
        --arch=*)
            arch="${arg#*=}"
            ;;
        --num-processes)
            python_args+=("$arg")
            if ((arg_idx < $#)); then
                ((arg_idx++))
                num_processes="${!arg_idx}"
                python_args+=("$num_processes")
            fi
            ;;
        --num-processes=*)
            num_processes="${arg#*=}"
            python_args+=("$arg")
            ;;
        -o|--output)
            if ((arg_idx < $#)); then
                ((arg_idx++))
                output_dir="${!arg_idx}"
            fi
            ;;
        --output=*)
            output_dir="${arg#*=}"
            ;;
        -h|--help)
            sed -n '3,17p' "$0"
            exit 0
            ;;
        *)
            python_args+=("$arg")
            ;;
    esac
done

# Resolve arch -> kernel name + test script + default num-processes.
case "$arch" in
    sm100)
        kernel_name=sm100_fp8_fp4_mega_moe_impl
        test_script=tests/test_mega_moe.py
        # SM100 path uses dispatch+combine via DeepEP; default 8 ranks.
        : "${num_processes:=8}"
        # SM100 test does not use --benchmark; it has --ncu-profile-only directly.
        bench_flag=()
        ;;
    sm90)
        kernel_name=sm90_fp8_mega_moe_impl
        test_script=tests/test_mega_moe_sm90.py
        # SM90 default: 2 ranks (matches test default).
        : "${num_processes:=2}"
        # SM90 NCU mode: --benchmark gates the perf path; --ncu-profile-only
        # short-circuits to a single kernel launch.
        bench_flag=(--benchmark)
        ;;
    "")
        echo "ERROR: must pass --arch sm100|sm90" >&2
        exit 2
        ;;
    *)
        echo "ERROR: unknown --arch '$arch' (expected sm100 or sm90)" >&2
        exit 2
        ;;
esac

# If the user did not pass --num-processes via CLI, append our default so the
# Python script and ncu --communicator-tcp-num-peers stay in sync.
if [[ ! " ${python_args[*]} " =~ " --num-processes " && \
      ! " ${python_args[*]} " =~ " --num-processes=" ]]; then
    python_args+=(--num-processes "$num_processes")
fi

echo "Arch: $arch"
echo "Kernel: $kernel_name"
echo "Test script: $test_script"
echo "Python Args: ${python_args[*]}"
echo "Num Processes: $num_processes"
echo "Output Dir: $output_dir"
mkdir -p "$output_dir"

# All NCU child processes need to agree on rendezvous. Pick one free port now
# and export it so each `--local-rank-idx=$i` invocation sees the same value.
export MASTER_ADDR=127.0.0.1
if [[ -z "${MASTER_PORT:-}" ]]; then
    # Probe a free TCP port. Python is the most portable way.
    MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
    export MASTER_PORT
fi
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"

export DG_JIT_WITH_LINEINFO=1 # for source counters

echo "Warm up JIT cache"
python "$test_script" --ncu-profile-only "${bench_flag[@]}" "${python_args[@]}"

sleep 2

ncu_args=(
    --config-file off
    --force-overwrite
    --kernel-name "$kernel_name"
    --import-source yes
    --replay-mode application
    --section PmSampling
    --section SourceCounters
    --rule LocalMemoryUsage
    --launch-skip 0
    --launch-count 1
    --lockstep-kernel-launch
    --communicator tcp
    --clock-control none
    --pm-sampling-interval 1000
    --pm-sampling-max-passes 1
    --disable-pm-warp-sampling
    --communicator-tcp-num-peers "$num_processes"
    --kill yes
    --app-replay-buffer memory
)

echo "Run Job"

for ((i = 0; i < num_processes; ++i)); do
    ncu ${ncu_args[@]} -o "${output_dir%/}/mega-moe.${arch}.$i" \
        python "$test_script" \
            --local-rank-idx=$i \
            --ncu-profile-only \
            "${bench_flag[@]}" \
            "${python_args[@]}" &
done

echo "Waiting"
wait
echo "Done"
