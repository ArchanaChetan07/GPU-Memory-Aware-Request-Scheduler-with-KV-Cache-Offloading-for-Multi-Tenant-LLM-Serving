"""
Dispatch layer for Request-Level Context Swapping.

Routes swap operations to CUDA kernels when available, falling back to
NumPy reference implementations otherwise.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .context_swap_ref import ContextSwapper

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from . import _C
    HAS_CUDA = _C is not None
except ImportError:
    HAS_CUDA = False
    _C = None

if not HAS_CUDA:
    # Opt-in JIT compile: CONTEXT_SWAP_JIT_CUDA=1 (needs GPU + nvcc)
    from ._jit import load_extension
    _C = load_extension("context_swapping_C", "CONTEXT_SWAP_JIT_CUDA")
    HAS_CUDA = _C is not None


def _gpu_available() -> bool:
    """GPU swap path needs torch.cuda; the compiled ext is optional
    (a torch pinned-memory fallback covers the same operations)."""
    return HAS_TORCH and torch.cuda.is_available()


def _use_cuda(force: Optional[bool] = None) -> bool:
    if force is not None:
        return force and _gpu_available()
    return _gpu_available()


class GPUContextSwapper:
    """GPU-backed context swapper using pinned memory + async streams.

    Falls back to the NumPy reference swapper when CUDA is unavailable.
    """

    def __init__(
        self,
        total_blocks: int = 4096,
        block_numel: int = 16384,     # e.g. 16 tokens x 8 heads x 128 dim
        staging_blocks: int = 256,
        buffer_capacity_mb: int = 8192,
    ):
        self.total_blocks = total_blocks
        self.block_numel = block_numel
        self.staging_blocks = staging_blocks
        self.cuda_active = _use_cuda()

        if self.cuda_active:
            self.kv_cache = torch.zeros(
                (total_blocks, block_numel), dtype=torch.float32, device='cuda')
            self.gpu_staging = torch.zeros(
                (staging_blocks, block_numel), dtype=torch.float32, device='cuda')
            self.pinned_pool: Dict[str, torch.Tensor] = {}
            self.saved_indices: Dict[str, torch.Tensor] = {}
        else:
            self.ref_swapper = ContextSwapper(buffer_capacity_mb=buffer_capacity_mb)

    def swap_out(self, request_id: str, block_indices: List[int]) -> bool:
        """Evict blocks for a request to pinned host memory.

        CPU fallback limitation: without a GPU there is no kv_cache tensor
        to gather from, so the reference path only exercises the
        bookkeeping (allocation, metadata, restore ordering) with
        placeholder data — it does NOT preserve real KV contents.
        """
        if not self.cuda_active:
            kv_blocks = {i: np.zeros((8, self.block_numel // 8), dtype=np.float32)
                         for i in block_indices}
            return self.ref_swapper.save_context(
                request_id, kv_blocks, block_indices, current_position=0)

        n = len(block_indices)
        assert n <= self.staging_blocks, "too many blocks for staging buffer"

        idx = torch.tensor(block_indices, dtype=torch.int32, device='cuda')
        pinned = torch.empty((n, self.block_numel), dtype=torch.float32,
                             pin_memory=True)

        if _C is not None:
            _C.swap_out(self.kv_cache, idx, self.gpu_staging[:n], pinned)
        else:
            # torch fallback: still uses pinned memory + async copy
            gathered = self.kv_cache[idx.long()]
            pinned.copy_(gathered, non_blocking=True)

        torch.cuda.current_stream().synchronize()
        self.pinned_pool[request_id] = pinned
        self.saved_indices[request_id] = idx
        return True

    def swap_in(self, request_id: str) -> bool:
        """Restore blocks for a request from pinned host memory."""
        if not self.cuda_active:
            return self.ref_swapper.restore_context(request_id) is not None

        if request_id not in self.pinned_pool:
            return False

        pinned = self.pinned_pool.pop(request_id)
        idx = self.saved_indices.pop(request_id)
        n = idx.shape[0]

        if _C is not None:
            _C.swap_in(pinned, idx, self.gpu_staging[:n], self.kv_cache)
        else:
            staged = pinned.to('cuda', non_blocking=True)
            self.kv_cache[idx.long()] = staged

        torch.cuda.current_stream().synchronize()
        return True

    def num_suspended(self) -> int:
        if self.cuda_active:
            return len(self.pinned_pool)
        return len(self.ref_swapper.buffer.metadata_map)


def backend_status() -> dict:
    """Return current backend availability."""
    return {
        'has_torch': HAS_TORCH,
        'has_cuda': HAS_CUDA,
        'active_backend': 'cuda' if _use_cuda() else 'reference',
    }
