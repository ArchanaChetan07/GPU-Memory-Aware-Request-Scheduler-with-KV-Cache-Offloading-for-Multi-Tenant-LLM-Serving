# Contributing

## Development setup

```bash
pip install -e ".[dev]"
pytest tests/ -q                       # scheduler/queue/swap bookkeeping tests
python scripts/simulate_workload.py    # admission simulation
```

With a CUDA GPU the pinned-memory swap path runs via the torch fallback; with
nvcc the compiled kernels JIT-build (PowerShell/cmd on Windows):

```powershell
$env:CONTEXT_SWAP_JIT_CUDA = "1"
pytest tests/ -q
python benchmarks/bench_context_swap.py
```

## The one rule: bit-exact swaps

A context swap that corrupts KV data is worse than useless. Any change to the
swap path must keep the round-trip tests passing bit-exact — both the NumPy
reference (`test_context_swap.py`) and the GPU path (`TestGPUSwap`).

## Guidelines

- Scheduler changes must preserve deterministic ordering (the `request_id`
  tiebreaker in priority tuples) and the SLA-protection guard on victim selection
- Benchmarks that back README claims are committed to `results/` as JSON
- Measured numbers and design targets stay clearly separated in docs

## Reporting issues

Include your GPU model, CUDA/PyTorch versions, and whether the failure occurs
on the reference swapper, the torch fallback, or the compiled-kernel path.
