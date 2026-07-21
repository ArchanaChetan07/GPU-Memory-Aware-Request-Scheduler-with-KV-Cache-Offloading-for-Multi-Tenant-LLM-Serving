"""Test suite for Priority Queue

Week 2 deliverable: Priority queue correctness tests
"""

import pytest
from src.priority_queue_ref import (
    PriorityQueue,
    SchedulingStrategy,
    compute_priority_tuple,
)


class TestPriorityQueueBasic:
    """Basic priority queue operations"""

    def test_push_pop_single(self):
        """Test single push/pop"""
        queue = PriorityQueue()
        queue.push("req_1", (100, -10, "req_1"), payload={"data": "value"})

        req_id, payload = queue.pop()
        assert req_id == "req_1"
        assert payload["data"] == "value"

    def test_push_pop_multiple(self):
        """Test multiple push/pop maintains order"""
        queue = PriorityQueue()
        queue.push("req_3", (300, -1, "req_3"))
        queue.push("req_1", (100, -1, "req_1"))
        queue.push("req_2", (200, -1, "req_2"))

        assert queue.pop()[0] == "req_1"
        assert queue.pop()[0] == "req_2"
        assert queue.pop()[0] == "req_3"

    def test_peek(self):
        """Test peeking without removal"""
        queue = PriorityQueue()
        queue.push("req_1", (100, 0, "req_1"))
        queue.push("req_2", (50, 0, "req_2"))

        assert queue.peek() == "req_2"
        assert queue.peek() == "req_2"  # Still there
        assert queue.pop()[0] == "req_2"

    def test_empty_queue(self):
        """Test operations on empty queue"""
        queue = PriorityQueue()

        assert queue.peek() is None
        assert queue.pop() is None
        assert len(queue) == 0
        assert not queue


class TestPriorityQueueLazyDeletion:
    """Test lazy deletion mechanism"""

    def test_remove_request(self):
        """Test removing request from queue"""
        queue = PriorityQueue()
        queue.push("req_1", (100, 0, "req_1"))
        queue.push("req_2", (50, 0, "req_2"))

        removed = queue.remove("req_1")
        assert removed

        assert queue.pop()[0] == "req_2"
        assert queue.pop() is None

    def test_remove_nonexistent(self):
        """Test removing non-existent request"""
        queue = PriorityQueue()
        queue.push("req_1", (100, 0, "req_1"))

        removed = queue.remove("req_nonexistent")
        assert not removed

    def test_lazy_deletion_efficiency(self):
        """Test that lazy deletion doesn't re-add removed items"""
        queue = PriorityQueue()

        for i in range(100):
            queue.push(f"req_{i}", (float(i), 0, f"req_{i}"))

        # Remove half
        for i in range(0, 50):
            queue.remove(f"req_{i}")

        # Pop should skip removed
        count = 0
        while queue.pop():
            count += 1

        assert count == 50


class TestPriorityTupleComputation:
    """Test priority tuple computation"""

    def test_sla_first_strategy(self):
        """Test SLA-first priority tuple"""
        priority = compute_priority_tuple(
            "req_1",
            sla_remaining_ms=100.0,
            revenue_value=10.0,
            strategy=SchedulingStrategy.SLA_FIRST
        )

        # Lower SLA should have higher priority
        priority_urgent = compute_priority_tuple(
            "req_urgent",
            sla_remaining_ms=50.0,
            revenue_value=10.0,
            strategy=SchedulingStrategy.SLA_FIRST
        )

        assert priority_urgent < priority

    def test_revenue_first_strategy(self):
        """Test revenue-first priority tuple"""
        priority_low = compute_priority_tuple(
            "req_low",
            sla_remaining_ms=100.0,
            revenue_value=1.0,
            strategy=SchedulingStrategy.REVENUE_FIRST
        )

        priority_high = compute_priority_tuple(
            "req_high",
            sla_remaining_ms=100.0,
            revenue_value=10.0,
            strategy=SchedulingStrategy.REVENUE_FIRST
        )

        # Higher revenue should have higher priority
        assert priority_high < priority_low

    def test_hybrid_strategy(self):
        """Test hybrid strategy balances both metrics"""
        priority_1 = compute_priority_tuple(
            "req_1",
            sla_remaining_ms=100.0,
            revenue_value=10.0,
            strategy=SchedulingStrategy.HYBRID
        )

        priority_2 = compute_priority_tuple(
            "req_2",
            sla_remaining_ms=50.0,
            revenue_value=5.0,
            strategy=SchedulingStrategy.HYBRID
        )

        # Both should produce valid tuples
        assert isinstance(priority_1, tuple)
        assert isinstance(priority_2, tuple)


class TestPriorityQueueSize:
    """Test size tracking"""

    def test_len(self):
        """Test queue length"""
        queue = PriorityQueue()

        for i in range(10):
            queue.push(f"req_{i}", (float(i), 0, f"req_{i}"))

        assert len(queue) == 10

    def test_len_after_pop(self):
        """Test length after pop"""
        queue = PriorityQueue()
        queue.push("req_1", (100, 0, "req_1"))
        queue.push("req_2", (50, 0, "req_2"))

        assert len(queue) == 2
        queue.pop()
        assert len(queue) == 1

    def test_len_after_remove(self):
        """Test length tracking with lazy deletion"""
        queue = PriorityQueue()
        queue.push("req_1", (100, 0, "req_1"))

        queue.remove("req_1")
        # Size hint still counts it
        assert len(queue) == 1
        # But pop returns None
        assert queue.pop() is None


class TestPriorityQueueClear:
    """Test clear operation"""

    def test_clear(self):
        """Test clearing queue"""
        queue = PriorityQueue()

        for i in range(10):
            queue.push(f"req_{i}", (float(i), 0, f"req_{i}"))

        queue.clear()
        assert len(queue) == 0
        assert queue.pop() is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
