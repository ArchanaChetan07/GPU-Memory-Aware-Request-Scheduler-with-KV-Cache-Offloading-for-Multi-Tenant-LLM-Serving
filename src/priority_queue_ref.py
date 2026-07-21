"""Priority Queue Reference Implementation

Heap-based priority queue for request scheduling.
"""

import heapq
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class SchedulingStrategy(Enum):
    """Scheduling strategy options"""
    SLA_FIRST = "sla_first"
    REVENUE_FIRST = "revenue_first"
    HYBRID = "hybrid"
    FAIR = "fair"


@dataclass
class QueueEntry:
    """Entry in priority queue"""
    priority: Tuple  # Tuple for heap comparison
    request_id: str
    payload: Any = None

    def __lt__(self, other: 'QueueEntry') -> bool:
        """For heap comparison"""
        if self.priority == other.priority:
            return self.request_id < other.request_id
        return self.priority < other.priority


class PriorityQueue:
    """Min-heap priority queue for request scheduling

    Supports different scheduling strategies via pluggable priority functions.
    """

    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.SLA_FIRST):
        self.strategy = strategy
        self.heap: List[QueueEntry] = []
        self.removed: set = set()  # Lazy deletion tracking
        self.active_ids: set = set()  # IDs currently live in the queue
        self.size_hint = 0

    def push(self, request_id: str, priority: Tuple, payload: Any = None) -> None:
        """Add request with priority tuple"""
        entry = QueueEntry(priority=priority, request_id=request_id, payload=payload)
        heapq.heappush(self.heap, entry)
        self.active_ids.add(request_id)
        self.size_hint += 1

    def pop(self) -> Optional[Tuple[str, Any]]:
        """Remove and return highest-priority request

        Returns (request_id, payload) or None if empty.
        """
        while self.heap:
            entry = heapq.heappop(self.heap)
            if entry.request_id not in self.removed:
                self.active_ids.discard(entry.request_id)
                self.size_hint -= 1
                return (entry.request_id, entry.payload)
            self.removed.discard(entry.request_id)
            self.active_ids.discard(entry.request_id)
            self.size_hint -= 1

        return None

    def peek(self) -> Optional[str]:
        """Peek at highest-priority request without removing"""
        while self.heap:
            entry = self.heap[0]
            if entry.request_id not in self.removed:
                return entry.request_id
            heapq.heappop(self.heap)
            self.removed.discard(entry.request_id)
            self.active_ids.discard(entry.request_id)

        return None

    def remove(self, request_id: str) -> bool:
        """Mark request for removal (lazy deletion)

        Returns False if the request is not currently in the queue.
        """
        if request_id not in self.active_ids or request_id in self.removed:
            return False
        self.removed.add(request_id)
        return True

    def __len__(self) -> int:
        """Number of active entries"""
        return self.size_hint

    def __bool__(self) -> bool:
        """True if not empty"""
        return self.size_hint > 0

    def clear(self) -> None:
        """Clear queue"""
        self.heap.clear()
        self.removed.clear()
        self.active_ids.clear()
        self.size_hint = 0


def compute_priority_tuple(
    request_id: str,
    sla_remaining_ms: float,
    revenue_value: float,
    strategy: SchedulingStrategy,
) -> Tuple:
    """Compute priority tuple based on strategy

    Lower tuples have higher priority in min-heap.
    """
    if strategy == SchedulingStrategy.SLA_FIRST:
        return (sla_remaining_ms, -revenue_value, request_id)

    elif strategy == SchedulingStrategy.REVENUE_FIRST:
        return (-revenue_value, sla_remaining_ms, request_id)

    elif strategy == SchedulingStrategy.HYBRID:
        sla_norm = max(sla_remaining_ms, 1.0)
        revenue_norm = max(revenue_value, 0.01)
        hybrid_score = 0.4 * sla_norm - 0.6 * revenue_norm
        return (hybrid_score, request_id)

    else:  # FAIR
        # Age-based FIFO
        return (0, request_id)  # Age would be tracked externally


class QueueBench:
    """Benchmark for priority queue operations"""

    @staticmethod
    def benchmark_push_pop(n: int = 10000, strategy: SchedulingStrategy = SchedulingStrategy.SLA_FIRST) -> dict:
        """Benchmark push/pop operations"""
        import time

        queue = PriorityQueue(strategy=strategy)

        # Push phase
        start = time.time()
        for i in range(n):
            priority = (float(i % 100), -float(i % 50))
            queue.push(f"req_{i}", priority)
        push_time = time.time() - start

        # Pop phase
        start = time.time()
        while queue:
            queue.pop()
        pop_time = time.time() - start

        return {
            'n': n,
            'strategy': strategy.value,
            'push_time_s': push_time,
            'pop_time_s': pop_time,
            'push_ops_per_sec': n / push_time,
            'pop_ops_per_sec': n / pop_time,
        }
