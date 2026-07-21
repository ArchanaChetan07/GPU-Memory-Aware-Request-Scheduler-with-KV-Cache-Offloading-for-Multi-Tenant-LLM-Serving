"""
Benchmark suite for Context Swapping.

Measures swap-out/swap-in latency on the active backend.
Target: <2ms per swap operation for typical request context sizes.
Writes results to results/*.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ops
from src.context_swap_ref import ContextSwapper


def bench_reference_swap(num_blocks: int = 32, iters: int = 50) -> dict:
    """Benchmark the NumPy reference swapper (upper bound on real latency)."""
    num_heads, head_dim = 8, 128
    swapper = ContextSwapper(buffer_capacity_mb=2048)

    save_times = []
    restore_times = []

    for i in range(iters):
        rid = f"req_{i}"
        kv_blocks = {
            j: np.random.randn(num_heads, head_dim).astype(np.float32)
            for j in range(num_blocks)
        }
        block_table = list(range(num_blocks))

        t0 = time.perf_counter()
        swapper.save_context(rid, kv_blocks, block_table, current_position=0)
        save_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        swapper.restore_context(rid)
        restore_times.append((time.perf_counter() - t0) * 1000)

    return {
        'op': 'reference_swap',
        'num_blocks': num_blocks,
        'block_shape': [num_heads, head_dim],
        'avg_save_ms': round(float(np.mean(save_times)), 4),
        'avg_restore_ms': round(float(np.mean(restore_times)), 4),
        'p99_save_ms': round(float(np.percentile(save_times, 99)), 4),
        'p99_restore_ms': round(float(np.percentile(restore_times, 99)), 4),
    }


def bench_gpu_swap(num_blocks: int = 64, block_numel: int = 16384, iters: int = 30) -> dict:
    """Benchmark GPU swap path with pinned memory (torch fallback if no ext)."""
    status = ops.backend_status()
    if status['active_backend'] != 'cuda':
        return {'op': 'gpu_swap', 'skipped': True, 'reason': 'CUDA unavailable'}

    import torch
    swapper = ops.GPUContextSwapper(
        total_blocks=1024, block_numel=block_numel, staging_blocks=num_blocks)

    out_times = []
    in_times = []
    indices = list(range(num_blocks))

    for i in range(iters):
        rid = f"req_{i}"

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        swapper.swap_out(rid, indices)
        out_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        swapper.swap_in(rid)
        in_times.append((time.perf_counter() - t0) * 1000)

    bytes_moved = num_blocks * block_numel * 4
    return {
        'op': 'gpu_swap',
        'num_blocks': num_blocks,
        'mb_per_swap': round(bytes_moved / (1024 * 1024), 1),
        'avg_swap_out_ms': round(float(np.mean(out_times)), 4),
        'avg_swap_in_ms': round(float(np.mean(in_times)), 4),
        'p99_swap_out_ms': round(float(np.percentile(out_times, 99)), 4),
        'p99_swap_in_ms': round(float(np.percentile(in_times, 99)), 4),
        'target_2ms_pass': bool(np.mean(out_times) < 2.0 and np.mean(in_times) < 2.0),
    }


def main():
    print("=" * 60)
    print("Context Swapping - Benchmark Suite")
    print("=" * 60)

    status = ops.backend_status()
    print(f"Backend: {status['active_backend']}\n")

    results = {'backend': status['active_backend'], 'benchmarks': []}

    ref = bench_reference_swap()
    results['benchmarks'].append(ref)
    print(f"Reference save:    {ref['avg_save_ms']:.3f} ms avg / {ref['p99_save_ms']:.3f} ms p99")
    print(f"Reference restore: {ref['avg_restore_ms']:.3f} ms avg / {ref['p99_restore_ms']:.3f} ms p99")

    gpu = bench_gpu_swap()
    results['benchmarks'].append(gpu)
    if gpu.get('skipped'):
        print(f"GPU swap: skipped ({gpu['reason']})")
    else:
        print(f"GPU swap-out ({gpu['mb_per_swap']} MB): {gpu['avg_swap_out_ms']:.3f} ms avg")
        print(f"GPU swap-in  ({gpu['mb_per_swap']} MB): {gpu['avg_swap_in_ms']:.3f} ms avg")
        print(f"<2ms target: {'PASS' if gpu['target_2ms_pass'] else 'FAIL'}")

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, 'swap_latency.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == '__main__':
    main()
