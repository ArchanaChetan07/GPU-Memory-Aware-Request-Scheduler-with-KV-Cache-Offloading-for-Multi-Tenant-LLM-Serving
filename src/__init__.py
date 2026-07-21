"""Request-Level Context Swapping

Adaptive multi-tenant request scheduling with dynamic context eviction
for GPU-constrained LLM serving.

Components:
- scheduler_ref.py: Request scheduler (Week 2)
- priority_queue_ref.py: Priority-based scheduling (Week 2)
- context_swap_ref.py: Context save/restore (Week 2)
- csrc/context_swap_kernel.cu: CUDA kernel (Week 3-5)
- csrc/bindings.cpp: PyTorch integration (Week 3-5)

Expected impact:
- GPU utilization: 65% → 85%
- Concurrent requests: +40%
- Revenue per GPU: +$2,400/month
- P99 SLA violations: -60%
"""

__version__ = "0.1.0"

try:
    from .scheduler_ref import (
        RequestScheduler,
        Request,
    )
    HAS_SCHEDULER_REF = True
except ImportError:
    HAS_SCHEDULER_REF = False

try:
    from .priority_queue_ref import (
        PriorityQueue,
        SchedulingStrategy,
    )
    HAS_QUEUE_REF = True
except ImportError:
    HAS_QUEUE_REF = False

try:
    from .context_swap_ref import (
        ContextSwapper,
        ContextBuffer,
    )
    HAS_SWAP_REF = True
except ImportError:
    HAS_SWAP_REF = False

try:
    from . import _C
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    _C = None

__all__ = [
    'RequestScheduler',
    'Request',
    'PriorityQueue',
    'SchedulingStrategy',
    'ContextSwapper',
    'ContextBuffer',
    'HAS_SCHEDULER_REF',
    'HAS_QUEUE_REF',
    'HAS_SWAP_REF',
    'HAS_CUDA',
]

def backend_info():
    """Print available backends"""
    print(f"Scheduler reference: {'✓' if HAS_SCHEDULER_REF else '✗'}")
    print(f"Priority queue reference: {'✓' if HAS_QUEUE_REF else '✗'}")
    print(f"Context swap reference: {'✓' if HAS_SWAP_REF else '✗'}")
    print(f"CUDA extension: {'✓' if HAS_CUDA else '✗'}")
