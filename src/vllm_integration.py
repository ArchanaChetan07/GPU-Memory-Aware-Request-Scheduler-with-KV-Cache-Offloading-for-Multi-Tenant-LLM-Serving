"""
vLLM integration shim for Request-Level Context Swapping.

Bridges the priority scheduler and GPU context swapper into a single
admission-control surface that a vLLM engine loop can call each step.

Usage (inside a vLLM fork or plugin):

    from context_swapping.vllm_integration import SwappingEngine

    engine = SwappingEngine(max_active=64)
    engine.submit(request_id, revenue=5.0, sla_ms=500, block_indices=[...])
    batch = engine.step()           # request ids to run this iteration
    engine.finish(request_id)       # request completed
"""

from typing import Dict, List, Optional

from .scheduler_ref import RequestScheduler, SchedulingStrategy
from . import ops


class SwappingEngine:
    """Admission control + context swapping for a vLLM engine loop."""

    def __init__(
        self,
        max_active: int = 64,
        strategy: SchedulingStrategy = SchedulingStrategy.SLA_FIRST,
        total_blocks: int = 4096,
        block_numel: int = 16384,
        staging_blocks: int = 256,
    ):
        self.scheduler = RequestScheduler(
            strategy=strategy,
            max_active_requests=max_active,
        )
        self.swapper = ops.GPUContextSwapper(
            total_blocks=total_blocks,
            block_numel=block_numel,
            staging_blocks=staging_blocks,
        )
        self.block_map: Dict[str, List[int]] = {}  # request -> KV block indices

    def submit(
        self,
        request_id: str,
        revenue: float,
        sla_ms: float,
        block_indices: List[int],
        max_tokens: int = 2048,
    ) -> bool:
        """Register a request and try to admit it.

        If the batch is full, the scheduler picks a victim, whose context
        is swapped out to pinned memory before admission.
        """
        self.scheduler.add_request(request_id, revenue_value=revenue,
                                   sla_deadline_ms=sla_ms, max_tokens=max_tokens)
        self.block_map[request_id] = block_indices

        victim = None
        if len(self.scheduler.active_requests) >= self.scheduler.max_active_requests:
            victim = self.scheduler.select_swapout_candidate()

        admitted = self.scheduler.admit_request(request_id)

        # The scheduler suspended the victim; persist its context
        if admitted and victim is not None and \
                victim.request_id in self.scheduler.suspended_requests:
            self.swapper.swap_out(victim.request_id,
                                  self.block_map.get(victim.request_id, []))
        return admitted

    def step(self, batch_size: Optional[int] = None) -> List[str]:
        """Get the next batch, restoring suspended requests as slots free up."""
        if batch_size is None:
            batch_size = self.scheduler.max_active_requests

        # Restore contexts for any request the scheduler pulls back in
        before = set(self.scheduler.suspended_requests.keys())
        batch = self.scheduler.get_next_batch(batch_size)
        after = set(self.scheduler.suspended_requests.keys())

        for restored_id in before - after:
            self.swapper.swap_in(restored_id)

        return batch

    def finish(self, request_id: str) -> None:
        """Mark a request complete and release its resources."""
        self.scheduler.complete_request(request_id)
        self.block_map.pop(request_id, None)

    def stats(self) -> dict:
        s = self.scheduler.stats()
        s['suspended_contexts'] = self.swapper.num_suspended()
        return s
