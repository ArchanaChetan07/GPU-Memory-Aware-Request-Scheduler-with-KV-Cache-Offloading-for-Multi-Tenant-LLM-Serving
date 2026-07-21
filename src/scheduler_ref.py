"""Request Scheduler Reference Implementation

Request-level scheduling with priority-aware decisions for multi-tenant LLM serving.

Algorithm:
1. Maintain priority queue of active requests
2. When memory pressure: select lowest-priority victim
3. Swap out victim context to pinned memory
4. When batch slot available: restore highest-priority waiting request

Priority metric: (SLA_remaining, -revenue_value)
- Lexicographic: prioritize SLA-critical requests
- Secondary: high-revenue within same SLA window
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class SchedulingStrategy(Enum):
    """Scheduling strategy options"""
    SLA_FIRST = "sla_first"           # Prioritize SLA deadline
    REVENUE_FIRST = "revenue_first"   # Prioritize revenue
    HYBRID = "hybrid"                 # Balance SLA + revenue
    FAIR = "fair"                     # Fair queuing


@dataclass
class Request:
    """Represents a request in the scheduler"""
    request_id: str
    revenue_value: float          # Revenue in dollars
    sla_deadline_ms: float        # SLA budget from creation time (ms)
    current_tokens: int = 0       # Tokens generated so far
    max_tokens: int = 2048        # Max tokens to generate
    priority_weight: float = 1.0  # Weight in priority queue
    timestamp_created: float = field(default_factory=time.time)
    status: str = "pending"       # pending, active, suspended, completed

    def sla_remaining(self) -> float:
        """Time remaining to meet SLA (ms).

        Decreases as wall-clock time passes — a waiting request becomes
        progressively more urgent under SLA_FIRST ordering, and goes
        negative once its deadline is missed.
        """
        elapsed_ms = (time.time() - self.timestamp_created) * 1000.0
        return self.sla_deadline_ms - elapsed_ms

    def priority_tuple(self, strategy: SchedulingStrategy) -> Tuple:
        """Compute priority tuple for heap ordering

        Smaller tuples have higher priority.
        Used with heapq (min-heap).
        """
        if strategy == SchedulingStrategy.SLA_FIRST:
            # Prioritize: closer to deadline (lower remaining time)
            # Tiebreak: higher revenue
            return (self.sla_remaining(), -self.revenue_value, self.request_id)

        elif strategy == SchedulingStrategy.REVENUE_FIRST:
            # Prioritize: higher revenue
            # Tiebreak: closer to deadline
            return (-self.revenue_value, self.sla_remaining(), self.request_id)

        elif strategy == SchedulingStrategy.HYBRID:
            # Balanced: 40% SLA, 60% revenue
            sla_norm = max(self.sla_remaining(), 1.0)
            revenue_norm = max(self.revenue_value, 0.01)
            hybrid_score = 0.4 * sla_norm - 0.6 * revenue_norm
            return (hybrid_score, self.request_id)

        else:  # FAIR
            # First-come-first-served with age
            age_ms = (time.time() - self.timestamp_created) * 1000
            return (age_ms, self.request_id)

    def __lt__(self, other: 'Request') -> bool:
        """For heap comparison"""
        return self.request_id < other.request_id


class RequestScheduler:
    """Request scheduler with priority-aware swapping

    Manages active and suspended requests, makes swap-out/swap-in decisions.
    """

    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.SLA_FIRST,
        max_active_requests: int = 64,
        swap_threshold_percent: float = 80.0,
        memory_budget_mb: int = 40960,  # 40 GB
    ):
        self.strategy = strategy
        self.max_active_requests = max_active_requests
        self.swap_threshold_percent = swap_threshold_percent
        self.memory_budget_mb = memory_budget_mb

        # Tracking
        self.active_requests: Dict[str, Request] = {}
        self.suspended_requests: Dict[str, Request] = {}
        self.pending_requests: Dict[str, Request] = {}
        self.completed_requests: Dict[str, Request] = {}

        self.memory_usage_mb: float = 0.0
        self.total_requests_seen: int = 0

    def add_request(
        self,
        request_id: str,
        revenue_value: float,
        sla_deadline_ms: float,
        max_tokens: int = 2048,
    ) -> Request:
        """Add new request to pending queue"""
        request = Request(
            request_id=request_id,
            revenue_value=revenue_value,
            sla_deadline_ms=sla_deadline_ms,
            max_tokens=max_tokens,
        )
        self.pending_requests[request_id] = request
        self.total_requests_seen += 1
        return request

    def admit_request(self, request_id: str) -> bool:
        """Admit request from pending to active queue

        Returns True if admitted, False if memory insufficient.
        """
        if request_id not in self.pending_requests:
            return False

        if len(self.active_requests) >= self.max_active_requests:
            # Try to swap out a victim
            victim = self.select_swapout_candidate()
            if victim:
                self._suspend_request(victim.request_id)
            else:
                return False

        request = self.pending_requests.pop(request_id)
        request.status = "active"
        self.active_requests[request_id] = request

        # Estimate memory: ~100 bytes per token
        est_memory_mb = (request.max_tokens * 2 * 100) / (1024 * 1024)
        self.memory_usage_mb += est_memory_mb

        return True

    def select_swapout_candidate(self) -> Optional[Request]:
        """Select lowest-priority active request to evict

        Returns None if no candidate (e.g., all SLA-critical).
        """
        if not self.active_requests:
            return None

        # Find request with worst priority
        candidates = list(self.active_requests.values())
        candidates.sort(key=lambda r: r.priority_tuple(self.strategy), reverse=True)

        victim = candidates[0]

        # Safety: don't evict if it violates its SLA
        if victim.sla_remaining() < 100:  # < 100ms left
            return None

        return victim

    def _suspend_request(self, request_id: str):
        """Move request from active to suspended"""
        if request_id not in self.active_requests:
            return

        request = self.active_requests.pop(request_id)
        request.status = "suspended"
        self.suspended_requests[request_id] = request

        # Free memory
        est_memory_mb = (request.max_tokens * 2 * 100) / (1024 * 1024)
        self.memory_usage_mb = max(0.0, self.memory_usage_mb - est_memory_mb)

    def restore_request(self, request_id: str) -> bool:
        """Restore suspended request to active

        Returns False if SLA violated.
        """
        if request_id not in self.suspended_requests:
            return False

        request = self.suspended_requests[request_id]

        if request.sla_remaining() < 0:
            # SLA already violated
            return False

        self.suspended_requests.pop(request_id)
        request.status = "active"
        self.active_requests[request_id] = request

        est_memory_mb = (request.max_tokens * 2 * 100) / (1024 * 1024)
        self.memory_usage_mb += est_memory_mb

        return True

    def complete_request(self, request_id: str):
        """Mark request as completed"""
        for dict_key in [self.active_requests, self.suspended_requests, self.pending_requests]:
            if request_id in dict_key:
                request = dict_key.pop(request_id)
                request.status = "completed"
                self.completed_requests[request_id] = request

                est_memory_mb = (request.max_tokens * 2 * 100) / (1024 * 1024)
                self.memory_usage_mb = max(0.0, self.memory_usage_mb - est_memory_mb)
                break

    def should_swap_out(self) -> bool:
        """Check if memory pressure requires swap-out"""
        memory_percent = (self.memory_usage_mb / self.memory_budget_mb) * 100
        return memory_percent > self.swap_threshold_percent

    def reorder_batch(self, request_ids: List[str]) -> List[str]:
        """Reorder batch by priority

        Takes current active requests and reorders by priority for next batch.
        """
        requests_in_batch = [
            self.active_requests[rid] for rid in request_ids
            if rid in self.active_requests
        ]
        requests_in_batch.sort(key=lambda r: r.priority_tuple(self.strategy))
        return [r.request_id for r in requests_in_batch]

    def get_next_batch(self, batch_size: int) -> List[str]:
        """Get next batch of requests to process

        Prioritizes active, then can restore suspended if space.
        """
        batch = []

        # Add active requests by priority
        active_by_priority = sorted(
            self.active_requests.values(),
            key=lambda r: r.priority_tuple(self.strategy)
        )
        for req in active_by_priority:
            if len(batch) >= batch_size:
                break
            batch.append(req.request_id)

        # Try to restore suspended if space
        suspended_by_priority = sorted(
            self.suspended_requests.values(),
            key=lambda r: r.priority_tuple(self.strategy)
        )
        for req in suspended_by_priority:
            if len(batch) >= batch_size:
                break
            if self.restore_request(req.request_id):
                batch.append(req.request_id)

        # Try to admit pending if space
        pending_by_priority = sorted(
            self.pending_requests.values(),
            key=lambda r: r.priority_tuple(self.strategy)
        )
        for req in pending_by_priority:
            if len(batch) >= batch_size:
                break
            if self.admit_request(req.request_id):
                batch.append(req.request_id)

        return batch

    def memory_utilization(self) -> float:
        """Current memory utilization as percentage"""
        return (self.memory_usage_mb / self.memory_budget_mb) * 100

    def stats(self) -> dict:
        """Return scheduler statistics"""
        return {
            'active_requests': len(self.active_requests),
            'suspended_requests': len(self.suspended_requests),
            'pending_requests': len(self.pending_requests),
            'completed_requests': len(self.completed_requests),
            'memory_usage_mb': self.memory_usage_mb,
            'memory_utilization_percent': self.memory_utilization(),
            'total_requests_seen': self.total_requests_seen,
        }
