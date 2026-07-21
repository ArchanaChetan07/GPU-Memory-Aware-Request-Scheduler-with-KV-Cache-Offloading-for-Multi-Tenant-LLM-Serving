# Architecture: Request-Level Context Swapping

## Component Map

```
┌──────────────────────────────────────────────────────────┐
│                    vLLM Engine Loop                      │
└───────────────┬──────────────────────────────────────────┘
                │ submit / step / finish
┌───────────────▼──────────────────────────────────────────┐
│           src/vllm_integration.py                        │
│  SwappingEngine (admission control facade)               │
└──────┬──────────────────────────────┬────────────────────┘
       │                              │
┌──────▼───────────────┐  ┌───────────▼───────────────────┐
│ scheduler_ref.py     │  │ src/ops.py                    │
│ RequestScheduler     │  │ GPUContextSwapper             │
│ - priority tuples    │  │ - swap_out: gather + D2H      │
│ - admit / suspend /  │  │ - swap_in: H2D + scatter      │
│   restore / complete │  │ - pinned host pool            │
│ priority_queue_ref.py│  └───────────┬───────────────────┘
│ - lazy-deletion heap │              │
└──────────────────────┘  ┌───────────▼───────────────────┐
                          │ csrc/context_swap_kernel.cu   │
                          │ - gather_kv_blocks_kernel     │
                          │ - scatter_kv_blocks_kernel    │
                          │ - zero_kv_blocks_kernel       │
                          │ (torch pinned-mem fallback    │
                          │  covers same ops w/o ext)     │
                          └───────────────────────────────┘
```

## Priority model

```
priority = (sla_remaining_ms, -revenue_value, request_id)   # SLA_FIRST
```

Lexicographic min-heap ordering:
1. Requests closest to their SLA deadline run first.
2. Within the same deadline window, higher revenue wins.
3. `request_id` guarantees total (deterministic) ordering.

Alternate strategies (`REVENUE_FIRST`, `HYBRID` 40/60 blend, `FAIR`
age-based) are pluggable via `SchedulingStrategy`.

**SLA guard:** a request with < 100 ms of SLA budget left is never
selected as a swap-out victim (`select_swapout_candidate` returns None
rather than violate it).

## Swap mechanics

Swap-out (target < 2 ms, measured 0.95 ms avg for 4 MB on test HW):
1. `gather_kv_blocks_kernel` packs scattered cache blocks into a
   contiguous GPU staging buffer (coalesced reads).
2. `cudaMemcpyAsync` staging → pinned host on the current stream.
3. Pinned buffer parked in a per-request pool.

Swap-in (measured 0.52 ms avg for 4 MB):
1. `cudaMemcpyAsync` pinned host → GPU staging.
2. `scatter_kv_blocks_kernel` writes blocks back to their cache slots.

Why staging + gather/scatter instead of per-block copies: one large
contiguous PCIe transfer instead of N small ones — PCIe latency
(~10 µs/transfer) would otherwise dominate for typical 16-64 block
contexts.

## Admission flow (SwappingEngine.submit)

```
add_request(id, revenue, sla)
if batch full:
    victim = select_swapout_candidate()     # lowest priority, SLA-safe
admit_request(id)                           # scheduler suspends victim
if victim suspended:
    swapper.swap_out(victim)                # context -> pinned memory
```

`step()` mirrors this: `get_next_batch()` restores suspended requests
into freed slots and the engine swaps their contexts back in.

## Measured results (this repository)

| Metric | Baseline | With swapping |
|--------|----------|---------------|
| Admit rate (1000 reqs, 64 slots) | 31.3% | 100% |
| GPU swap-out (4 MB) | — | 0.95 ms avg |
| GPU swap-in (4 MB) | — | 0.52 ms avg |
| <2 ms target | — | PASS |

Reproduce: `python scripts/simulate_workload.py` and
`python benchmarks/bench_context_swap.py`.
