"""Integration tests for the vLLM shim (SwappingEngine)."""

import pytest

from src.vllm_integration import SwappingEngine
from src.scheduler_ref import SchedulingStrategy


def _make_engine(max_active=2):
    return SwappingEngine(
        max_active=max_active,
        total_blocks=64,
        block_numel=256,
        staging_blocks=16,
    )


class TestSwappingEngine:

    def test_submit_within_capacity(self):
        engine = _make_engine(max_active=4)
        assert engine.submit("r1", revenue=1.0, sla_ms=1000, block_indices=[0, 1])
        assert engine.submit("r2", revenue=2.0, sla_ms=1000, block_indices=[2, 3])

        stats = engine.stats()
        assert stats['active_requests'] == 2
        assert stats['suspended_contexts'] == 0

    def test_overflow_triggers_swap(self):
        """Submitting past capacity must suspend a low-priority victim."""
        engine = _make_engine(max_active=2)
        engine.submit("cheap", revenue=0.5, sla_ms=5000, block_indices=[0, 1])
        engine.submit("mid", revenue=2.0, sla_ms=2000, block_indices=[2, 3])

        # Third request: victim (lowest priority) gets swapped out
        assert engine.submit("premium", revenue=10.0, sla_ms=300, block_indices=[4, 5])

        stats = engine.stats()
        assert stats['active_requests'] == 2
        assert stats['suspended_requests'] == 1
        assert stats['suspended_contexts'] == 1

    def test_step_restores_suspended(self):
        """When capacity frees up, step() must swap suspended requests back in."""
        engine = _make_engine(max_active=2)
        engine.submit("a", revenue=0.5, sla_ms=5000, block_indices=[0])
        engine.submit("b", revenue=1.0, sla_ms=5000, block_indices=[1])
        engine.submit("c", revenue=5.0, sla_ms=500, block_indices=[2])

        assert engine.stats()['suspended_requests'] == 1

        # Complete an active request, freeing a slot
        active_ids = list(engine.scheduler.active_requests.keys())
        engine.finish(active_ids[0])

        batch = engine.step()
        stats = engine.stats()
        assert stats['suspended_requests'] == 0, "suspended request must be restored"
        assert stats['suspended_contexts'] == 0, "context must be swapped back in"
        assert len(batch) == 2

    def test_finish_cleans_up(self):
        engine = _make_engine(max_active=2)
        engine.submit("r1", revenue=1.0, sla_ms=1000, block_indices=[0])
        engine.finish("r1")

        stats = engine.stats()
        assert stats['active_requests'] == 0
        assert stats['completed_requests'] == 1
        assert "r1" not in engine.block_map

    def test_full_lifecycle_many_requests(self):
        """Stress: 20 requests through a 4-slot engine, all complete."""
        engine = _make_engine(max_active=4)

        submitted = []
        for i in range(20):
            rid = f"req_{i}"
            engine.submit(rid, revenue=float(i % 5), sla_ms=1000 + i * 100,
                          block_indices=[i % 32])
            submitted.append(rid)

            # Drain periodically
            if i % 3 == 2:
                for done in engine.step()[:2]:
                    engine.finish(done)

        # Drain remaining
        for _ in range(20):
            batch = engine.step()
            if not batch:
                break
            for rid in batch:
                engine.finish(rid)

        stats = engine.stats()
        assert stats['active_requests'] == 0
        assert stats['suspended_requests'] == 0
        assert stats['completed_requests'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
