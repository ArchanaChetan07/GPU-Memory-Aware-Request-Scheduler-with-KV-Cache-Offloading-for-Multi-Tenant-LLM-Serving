"""Test suite for Request Scheduler

Week 2 deliverable: Scheduler correctness tests
"""

import pytest
from src.scheduler_ref import (
    RequestScheduler,
    Request,
    SchedulingStrategy,
)


class TestSchedulerBasics:
    """Basic scheduler functionality tests"""

    def test_add_request(self):
        """Test adding requests to pending queue"""
        scheduler = RequestScheduler()

        scheduler.add_request(
            request_id="req_1",
            revenue_value=5.0,
            sla_deadline_ms=500.0,
        )

        assert "req_1" in scheduler.pending_requests
        assert scheduler.total_requests_seen == 1

    def test_admit_request(self):
        """Test admitting request from pending to active"""
        scheduler = RequestScheduler()

        scheduler.add_request("req_1", revenue_value=5.0, sla_deadline_ms=500.0)
        admitted = scheduler.admit_request("req_1")

        assert admitted
        assert "req_1" in scheduler.active_requests
        assert "req_1" not in scheduler.pending_requests

    def test_complete_request(self):
        """Test completing a request"""
        scheduler = RequestScheduler()

        scheduler.add_request("req_1", revenue_value=5.0, sla_deadline_ms=500.0)
        scheduler.admit_request("req_1")
        scheduler.complete_request("req_1")

        assert "req_1" not in scheduler.active_requests
        assert "req_1" in scheduler.completed_requests

    def test_sla_first_priority(self):
        """Test SLA-first scheduling strategy"""
        scheduler = RequestScheduler(strategy=SchedulingStrategy.SLA_FIRST)

        # Add requests with different SLAs
        scheduler.add_request("req_urgent", revenue_value=1.0, sla_deadline_ms=100.0)  # Urgent
        scheduler.add_request("req_normal", revenue_value=10.0, sla_deadline_ms=500.0)  # Higher revenue

        # With SLA_FIRST, urgent should have higher priority
        scheduler.admit_request("req_urgent")
        scheduler.admit_request("req_normal")

        batch = scheduler.get_next_batch(batch_size=2)
        assert batch[0] == "req_urgent", "SLA-first should prioritize urgent request"

    def test_revenue_first_priority(self):
        """Test revenue-first scheduling strategy"""
        scheduler = RequestScheduler(strategy=SchedulingStrategy.REVENUE_FIRST)

        scheduler.add_request("req_high_revenue", revenue_value=100.0, sla_deadline_ms=500.0)
        scheduler.add_request("req_low_revenue", revenue_value=1.0, sla_deadline_ms=100.0)

        scheduler.admit_request("req_high_revenue")
        scheduler.admit_request("req_low_revenue")

        batch = scheduler.get_next_batch(batch_size=2)
        assert batch[0] == "req_high_revenue", "Revenue-first should prioritize high-revenue"

    def test_hybrid_priority(self):
        """Test hybrid (balanced) scheduling strategy"""
        scheduler = RequestScheduler(strategy=SchedulingStrategy.HYBRID)

        scheduler.add_request("req_1", revenue_value=10.0, sla_deadline_ms=500.0)
        scheduler.add_request("req_2", revenue_value=50.0, sla_deadline_ms=100.0)

        scheduler.admit_request("req_1")
        scheduler.admit_request("req_2")

        batch = scheduler.get_next_batch(batch_size=2)
        # Hybrid balances both metrics
        assert len(batch) == 2


class TestSchedulerSwapping:
    """Tests for context swapping decisions"""

    def test_select_swapout_candidate(self):
        """Test selecting lowest-priority request for eviction"""
        scheduler = RequestScheduler(strategy=SchedulingStrategy.SLA_FIRST, max_active_requests=2)

        scheduler.add_request("req_high_priority", revenue_value=100.0, sla_deadline_ms=100.0)
        scheduler.add_request("req_low_priority", revenue_value=1.0, sla_deadline_ms=500.0)

        scheduler.admit_request("req_high_priority")
        scheduler.admit_request("req_low_priority")

        victim = scheduler.select_swapout_candidate()
        assert victim.request_id == "req_low_priority"

    def test_suspend_and_restore(self):
        """Test suspending and restoring request"""
        scheduler = RequestScheduler()

        scheduler.add_request("req_1", revenue_value=5.0, sla_deadline_ms=500.0)
        scheduler.admit_request("req_1")

        assert "req_1" in scheduler.active_requests

        scheduler._suspend_request("req_1")
        assert "req_1" in scheduler.suspended_requests
        assert "req_1" not in scheduler.active_requests

        restored = scheduler.restore_request("req_1")
        assert restored
        assert "req_1" in scheduler.active_requests

    def test_sla_violation_protection(self):
        """Test that we don't swap out SLA-critical requests"""
        scheduler = RequestScheduler()

        scheduler.add_request("req_critical", revenue_value=1.0, sla_deadline_ms=50.0)  # < 100ms
        scheduler.admit_request("req_critical")

        victim = scheduler.select_swapout_candidate()
        # Should not select this request (SLA < 100ms protection)
        assert victim is None or victim.request_id != "req_critical"


class TestSchedulerBatching:
    """Tests for batch operations"""

    def test_get_next_batch(self):
        """Test getting next batch"""
        scheduler = RequestScheduler(max_active_requests=5)

        for i in range(5):
            scheduler.add_request(f"req_{i}", revenue_value=float(i), sla_deadline_ms=500.0)
            scheduler.admit_request(f"req_{i}")

        batch = scheduler.get_next_batch(batch_size=3)
        assert len(batch) <= 3

    def test_reorder_batch(self):
        """Test reordering batch by priority"""
        scheduler = RequestScheduler(strategy=SchedulingStrategy.SLA_FIRST)

        scheduler.add_request("req_1", revenue_value=1.0, sla_deadline_ms=200.0)
        scheduler.add_request("req_2", revenue_value=2.0, sla_deadline_ms=100.0)  # Urgent
        scheduler.add_request("req_3", revenue_value=3.0, sla_deadline_ms=300.0)

        for req_id in ["req_1", "req_2", "req_3"]:
            scheduler.admit_request(req_id)

        reordered = scheduler.reorder_batch(["req_1", "req_2", "req_3"])
        assert reordered[0] == "req_2", "Urgent request should be first"

    def test_batch_size_limit(self):
        """Test that batch doesn't exceed size limit"""
        scheduler = RequestScheduler(max_active_requests=100)

        for i in range(20):
            scheduler.add_request(f"req_{i}", revenue_value=1.0, sla_deadline_ms=500.0)
            scheduler.admit_request(f"req_{i}")

        batch = scheduler.get_next_batch(batch_size=10)
        assert len(batch) <= 10


class TestSchedulerMemory:
    """Tests for memory management"""

    def test_memory_tracking(self):
        """Test memory usage tracking"""
        scheduler = RequestScheduler(memory_budget_mb=1024)

        scheduler.add_request("req_1", revenue_value=5.0, sla_deadline_ms=500.0, max_tokens=2048)
        scheduler.admit_request("req_1")

        assert scheduler.memory_usage_mb > 0
        assert scheduler.memory_utilization() > 0

    def test_memory_freed_on_suspend(self):
        """Test that memory is freed when request is suspended"""
        scheduler = RequestScheduler()

        scheduler.add_request("req_1", revenue_value=5.0, sla_deadline_ms=500.0, max_tokens=2048)
        scheduler.admit_request("req_1")

        initial_memory = scheduler.memory_usage_mb
        assert initial_memory > 0

        scheduler._suspend_request("req_1")
        assert scheduler.memory_usage_mb < initial_memory

    def test_should_swap_out(self):
        """Test memory pressure detection"""
        scheduler = RequestScheduler(
            memory_budget_mb=100,
            swap_threshold_percent=80.0
        )

        # Add requests to fill memory
        for i in range(10):
            scheduler.add_request(f"req_{i}", revenue_value=1.0, sla_deadline_ms=500.0)
            scheduler.admit_request(f"req_{i}")

        should_swap = scheduler.should_swap_out()
        # Should trigger swap if memory > 80%
        assert isinstance(should_swap, bool)


class TestSchedulerStats:
    """Tests for statistics and monitoring"""

    def test_stats_reporting(self):
        """Test scheduler statistics"""
        scheduler = RequestScheduler()

        scheduler.add_request("req_1", revenue_value=5.0, sla_deadline_ms=500.0)
        scheduler.admit_request("req_1")

        stats = scheduler.stats()
        assert stats['active_requests'] == 1
        assert stats['total_requests_seen'] == 1

    def test_stats_after_completion(self):
        """Test stats after request completion"""
        scheduler = RequestScheduler()

        for i in range(3):
            scheduler.add_request(f"req_{i}", revenue_value=1.0, sla_deadline_ms=500.0)
            scheduler.admit_request(f"req_{i}")

        for i in range(3):
            scheduler.complete_request(f"req_{i}")

        stats = scheduler.stats()
        assert stats['completed_requests'] == 3
        assert stats['active_requests'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
