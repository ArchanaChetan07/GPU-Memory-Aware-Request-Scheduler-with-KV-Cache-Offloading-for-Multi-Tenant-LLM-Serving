# Project 3: Request-Level Context Swapping — Getting Started

**Complete folder ready for 14-week implementation**

---

## 📋 What's Included

- ✅ README.md — Project overview
- ✅ Folder structure (src/, tests/, csrc/, etc.)
- ✅ Build configuration (setup.py, pyproject.toml, CMakeLists.txt)
- ✅ Package templates

**Not yet (Week 1):**
- Request scheduler reference (Week 2)
- Context swap logic (Week 2)
- Priority queue (Week 2)

---

## 🚀 Quick Start

```bash
cd GPU-Memory-Aware-Request-Scheduler-with-KV-Cache-Offloading-for-Multi-Tenant-LLM-Serving
pip install -e .
pytest tests/ -q          # 47 passed (GPU tests use torch fallback or skip)
```

With a CUDA GPU + nvcc (run from PowerShell/cmd on Windows, not Git Bash):

```powershell
$env:CONTEXT_SWAP_JIT_CUDA = "1"   # JIT-compiles csrc/ on import (cached)
pytest tests/ -q                    # exercises the compiled gather/scatter kernels
python benchmarks/bench_context_swap.py   # 4 MB swap: ~0.6 ms out / ~0.5 ms in
```

---

## 📖 Week-by-Week Plan

### Week 1–2: Scheduler & Priority Queue Reference
**Goal:** Implement request-level scheduling

**Deliverables:**
- [ ] `src/scheduler_ref.py` (NumPy)
- [ ] `src/priority_queue_ref.py` (NumPy)
- [ ] `src/context_swap_ref.py` (NumPy)
- [ ] 7 scheduler tests
- [ ] Correctness validated

**Key insight:** Priority tuple (SLA_remaining, -revenue)

### Week 3–5: Context Swap CUDA Kernel
**Goal:** Fast KV cache save/restore

**Deliverables:**
- [ ] `csrc/context_swap_kernel.cu` (CUDA kernel)
- [ ] `csrc/bindings.cpp` (PyTorch)
- [ ] Swap kernel tests passing
- [ ] <2ms swap latency verified

**Key insight:** Pinned memory + async streams

### Week 6–7: Benchmarking
**Goal:** Latency validation

**Deliverables:**
- [ ] Latency benchmarks (committed JSON)
- [ ] Utilization reports
- [ ] Performance tests passing

### Week 8–10: vLLM Integration
**Goal:** Multi-request batching ready

**Deliverables:**
- [ ] vLLM scheduler integration
- [ ] Batch reordering logic
- [ ] Integration tests passing

### Week 11: Simulation
**Goal:** Synthetic workload validation

**Deliverables:**
- [ ] Workload simulator (1000+ requests)
- [ ] SLA monitoring enabled
- [ ] Results: GPU util 85%+

### Week 12–14: Production Deployment
**Goal:** Production-ready code

**Deliverables:**
- [ ] Operations runbook
- [ ] Monitoring + alerting
- [ ] Deployment playbook

---

## 📊 Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| **Week 2** | Scheduler tests | 7/7 passing |
| **Week 5** | Swap latency | <2ms |
| **Week 7** | GPU utilization | 80%+ |
| **Week 11** | Concurrent requests | 3× baseline |
| **Week 11** | SLA violations | -60% vs baseline |

---

## 🎯 This Week

1. Understand request scheduling concepts
2. Review priority queue algorithms
3. Plan Week 2 implementation

Then read `Project3-Complete-Technical-Plan.md` (in related docs folder) for:
- Detailed scheduling algorithm
- CUDA specifications
- Code templates for Week 3+
- Workload simulation

---

## 📝 Core Concepts

### Priority Metric
```
priority = (deadline - current_time, -revenue_value)
```
- Lexicographic ordering: prioritize SLA-critical requests
- Secondary: prioritize high-revenue within same SLA deadline

### Swap-Out Decision
```
if memory_usage > SWAP_THRESHOLD:
    victim = scheduler.select_swapout_candidate()
    save_context(victim)
    free_gpu_memory()
```

### Swap-In Decision
```
if batch_slot_available and victim.deadline_safe:
    restore_context(victim)
    resume_generation()
```

---

**Status:** Ready for Week 1-2 planning  
**Next:** Implement scheduler reference  
**Timeline:** 14 weeks to production  
