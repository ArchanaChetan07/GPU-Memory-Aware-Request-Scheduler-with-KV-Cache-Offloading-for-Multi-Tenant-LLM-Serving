"""
Context swap tests: reference swapper round-trips and GPU swap path.

GPU tests are skipped when torch.cuda is unavailable.
"""

import numpy as np
import pytest

from src.context_swap_ref import ContextSwapper, ContextBuffer, simulate_context_swapping
from src import ops


class TestContextBufferAllocation:
    """Pinned-buffer allocation bookkeeping."""

    def test_allocate_and_get(self):
        buf = ContextBuffer(capacity_mb=16)
        view, meta = buf.allocate("req_1", 1024)
        assert view.shape[0] == 1024
        assert buf.get_context("req_1") is not None

    def test_deallocate(self):
        buf = ContextBuffer(capacity_mb=16)
        buf.allocate("req_1", 1024)
        assert buf.deallocate("req_1")
        assert buf.get_context("req_1") is None
        assert not buf.deallocate("req_missing")

    def test_capacity_exceeded(self):
        buf = ContextBuffer(capacity_mb=1)
        with pytest.raises(RuntimeError):
            buf.allocate("req_big", 2 * 1024 * 1024)

    def test_utilization(self):
        buf = ContextBuffer(capacity_mb=1)
        assert buf.utilization() == 0.0
        buf.allocate("req_1", 512 * 1024)
        assert buf.utilization() == pytest.approx(50.0, abs=1.0)


class TestReferenceSwapRoundTrip:
    """Save/restore must preserve data."""

    def test_save_restore_roundtrip(self):
        swapper = ContextSwapper(buffer_capacity_mb=64)
        num_heads, head_dim = 8, 64

        kv_blocks = {
            i: np.random.randn(num_heads, head_dim).astype(np.float32)
            for i in range(4)
        }
        block_table = [0, 1, 2, 3]

        assert swapper.save_context("req_1", kv_blocks, block_table, current_position=100)

        result = swapper.restore_context("req_1")
        assert result is not None
        restored, position = result
        assert position == 100
        assert set(restored.keys()) == set(block_table)
        # Data must be bit-exact — a save/restore that corrupts values is
        # worse than useless in production
        for i in block_table:
            np.testing.assert_array_equal(restored[i], kv_blocks[i]), f"block {i} corrupted"

    def test_restore_missing_returns_none(self):
        swapper = ContextSwapper(buffer_capacity_mb=16)
        assert swapper.restore_context("req_missing") is None

    def test_stats_tracking(self):
        swapper = ContextSwapper(buffer_capacity_mb=64)
        kv = {0: np.random.randn(8, 64).astype(np.float32)}

        swapper.save_context("req_1", kv, [0], 0)
        swapper.restore_context("req_1")

        stats = swapper.stats()
        assert stats['num_saves'] == 1
        assert stats['num_restores'] == 1
        assert stats['avg_save_time_ms'] >= 0.0

    def test_simulation_runs(self):
        stats = simulate_context_swapping(num_requests=20, avg_kv_size_mb=1.0)
        assert stats['saved_count'] > 0
        assert stats['num_requests'] == 20


@pytest.mark.skipif(not ops._gpu_available(), reason="torch.cuda unavailable")
class TestGPUSwap:
    """GPU swap path with pinned memory (torch fallback or compiled ext)."""

    def test_gpu_swap_roundtrip(self):
        import torch
        swapper = ops.GPUContextSwapper(
            total_blocks=64, block_numel=1024, staging_blocks=16)

        # Fill some cache blocks with known values
        indices = [3, 7, 11]
        for i in indices:
            swapper.kv_cache[i] = float(i)

        original = swapper.kv_cache[indices].clone()

        assert swapper.swap_out("req_1", indices)
        assert swapper.num_suspended() == 1

        # Corrupt GPU blocks, then restore
        for i in indices:
            swapper.kv_cache[i] = -1.0

        assert swapper.swap_in("req_1")
        assert swapper.num_suspended() == 0

        restored = swapper.kv_cache[indices]
        assert torch.allclose(restored, original), "swap round-trip corrupted data"

    def test_swap_in_missing_returns_false(self):
        swapper = ops.GPUContextSwapper(
            total_blocks=16, block_numel=256, staging_blocks=4)
        assert not swapper.swap_in("req_missing")

    def test_multiple_concurrent_suspensions(self):
        swapper = ops.GPUContextSwapper(
            total_blocks=64, block_numel=512, staging_blocks=8)

        for r in range(4):
            idx = [r * 2, r * 2 + 1]
            swapper.kv_cache[idx] = float(r + 1)
            assert swapper.swap_out(f"req_{r}", idx)

        assert swapper.num_suspended() == 4
        for r in range(4):
            assert swapper.swap_in(f"req_{r}")
        assert swapper.num_suspended() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
