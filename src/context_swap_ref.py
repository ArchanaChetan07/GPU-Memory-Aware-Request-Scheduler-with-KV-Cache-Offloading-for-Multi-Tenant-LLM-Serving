"""Context Swap Reference Implementation

Save and restore KV cache contexts for suspended requests.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class ContextMetadata:
    """Metadata for saved context"""
    request_id: str
    block_table: List[int]           # Indices of KV blocks
    current_position: int            # Current generation position
    num_kv_heads: int
    head_dim: int
    total_blocks: int
    saved_time: float = 0.0


class ContextBuffer:
    """Manages pinned host memory for contexts"""

    def __init__(self, capacity_mb: int = 8192):
        """Initialize context buffer

        Args:
            capacity_mb: Total capacity in MB
        """
        self.capacity_mb = capacity_mb
        self.capacity_bytes = capacity_mb * 1024 * 1024
        self.buffer = np.zeros(self.capacity_bytes, dtype=np.uint8)
        self.offset = 0
        self.metadata_map: Dict[str, ContextMetadata] = {}
        self.offsets: Dict[str, Tuple[int, int]] = {}  # request_id -> (offset, size)

    def allocate(self, request_id: str, size_bytes: int) -> Tuple[np.ndarray, ContextMetadata]:
        """Allocate space for context

        Args:
            request_id: Request identifier
            size_bytes: Size needed

        Returns:
            (buffer_view, metadata)
        """
        if self.offset + size_bytes > self.capacity_bytes:
            raise RuntimeError(f"Context buffer full: {self.offset + size_bytes} > {self.capacity_bytes}")

        view = self.buffer[self.offset:self.offset + size_bytes]
        self.offsets[request_id] = (self.offset, size_bytes)
        self.offset += size_bytes

        metadata = ContextMetadata(
            request_id=request_id,
            block_table=[],
            current_position=0,
            num_kv_heads=0,
            head_dim=0,
            total_blocks=0,
            saved_time=time.time(),
        )
        self.metadata_map[request_id] = metadata

        return view, metadata

    def deallocate(self, request_id: str) -> bool:
        """Free space for context

        Note: This is a simple implementation. Real one would use memory pooling.
        """
        if request_id not in self.offsets:
            return False

        self.offsets.pop(request_id, None)
        self.metadata_map.pop(request_id, None)
        return True

    def get_context(self, request_id: str) -> Optional[Tuple[np.ndarray, ContextMetadata]]:
        """Retrieve saved context"""
        if request_id not in self.offsets:
            return None

        offset, size = self.offsets[request_id]
        view = self.buffer[offset:offset + size]
        metadata = self.metadata_map[request_id]
        return (view, metadata)

    def utilization(self) -> float:
        """Context buffer utilization as percentage"""
        return (self.offset / self.capacity_bytes) * 100


class ContextSwapper:
    """Handles context swapping (save/restore)"""

    def __init__(self, buffer_capacity_mb: int = 8192):
        self.buffer = ContextBuffer(capacity_mb=buffer_capacity_mb)
        self.swap_stats = {
            'num_saves': 0,
            'num_restores': 0,
            'total_save_time_ms': 0.0,
            'total_restore_time_ms': 0.0,
        }

    def save_context(
        self,
        request_id: str,
        kv_blocks: Dict[int, np.ndarray],  # block_idx -> KV data
        block_table: List[int],
        current_position: int,
    ) -> bool:
        """Save KV cache context to pinned memory

        Args:
            request_id: Request ID
            kv_blocks: Dictionary of block_idx -> KV array
            block_table: List of block indices in use
            current_position: Current generation position

        Returns:
            True if successful
        """
        start = time.time()

        # Compute total size needed
        total_size = sum(kv.nbytes for kv in kv_blocks.values())

        try:
            buffer_view, metadata = self.buffer.allocate(request_id, total_size)
        except RuntimeError:
            return False

        # Copy raw float32 bytes to buffer (view, not astype — astype would
        # truncate float values to integers and destroy the data)
        saved_blocks = []
        offset = 0
        kv = None
        for block_idx in block_table:
            if block_idx not in kv_blocks:
                continue

            kv = np.ascontiguousarray(kv_blocks[block_idx], dtype=np.float32)
            kv_bytes = kv.view(np.uint8).reshape(-1)
            buffer_view[offset:offset + kv_bytes.size] = kv_bytes
            offset += kv_bytes.size
            saved_blocks.append(block_idx)

        if kv is None:
            # Nothing to save; release the allocation
            self.buffer.deallocate(request_id)
            return False

        # Update metadata (block shapes are uniform within a request)
        metadata.block_table = saved_blocks
        metadata.current_position = current_position
        metadata.num_kv_heads = kv.shape[0] if kv.ndim >= 2 else 0
        metadata.head_dim = kv.shape[1] if kv.ndim >= 2 else 0
        metadata.total_blocks = len(saved_blocks)

        elapsed = time.time() - start
        self.swap_stats['num_saves'] += 1
        self.swap_stats['total_save_time_ms'] += elapsed * 1000

        return True

    def restore_context(
        self,
        request_id: str,
    ) -> Optional[Tuple[Dict[int, np.ndarray], int]]:
        """Restore KV cache context from pinned memory

        Args:
            request_id: Request ID

        Returns:
            (kv_blocks, current_position) or None if not found
        """
        start = time.time()

        context = self.buffer.get_context(request_id)
        if not context:
            return None

        buffer_view, metadata = context

        # Reconstruct blocks from buffer
        kv_blocks = {}
        offset = 0
        num_kv_heads = metadata.num_kv_heads
        head_dim = metadata.head_dim

        for block_idx in metadata.block_table:
            # Each block: (num_kv_heads, head_dim) - reconstruct as FP32
            block_size = num_kv_heads * head_dim * 4  # 4 bytes per float32
            block_data = buffer_view[offset:offset + block_size].copy()
            kv = np.frombuffer(block_data.tobytes(), dtype=np.float32).reshape((num_kv_heads, head_dim))
            kv_blocks[block_idx] = kv
            offset += block_size

        elapsed = time.time() - start
        self.swap_stats['num_restores'] += 1
        self.swap_stats['total_restore_time_ms'] += elapsed * 1000

        # Clean up
        self.buffer.deallocate(request_id)

        return (kv_blocks, metadata.current_position)

    def swap_time_ms(self, operation: str) -> float:
        """Average swap time in milliseconds

        Args:
            operation: 'save' or 'restore'

        Returns:
            Average time in milliseconds
        """
        if operation == 'save':
            if self.swap_stats['num_saves'] == 0:
                return 0.0
            return self.swap_stats['total_save_time_ms'] / self.swap_stats['num_saves']
        else:  # restore
            if self.swap_stats['num_restores'] == 0:
                return 0.0
            return self.swap_stats['total_restore_time_ms'] / self.swap_stats['num_restores']

    def stats(self) -> dict:
        """Return swap statistics"""
        return {
            'num_saves': self.swap_stats['num_saves'],
            'num_restores': self.swap_stats['num_restores'],
            'avg_save_time_ms': self.swap_time_ms('save'),
            'avg_restore_time_ms': self.swap_time_ms('restore'),
            'buffer_utilization_percent': self.buffer.utilization(),
        }


def simulate_context_swapping(
    num_requests: int = 100,
    avg_kv_size_mb: float = 50.0,
) -> dict:
    """Simulate context swapping workload

    Args:
        num_requests: Number of requests to simulate
        avg_kv_size_mb: Average KV size per request in MB

    Returns:
        Statistics dictionary
    """
    swapper = ContextSwapper(buffer_capacity_mb=int(num_requests * avg_kv_size_mb * 1.5))
    request_id_counter = 0

    saved_count = 0
    restored_count = 0

    for i in range(num_requests):
        request_id = f"req_{request_id_counter}"
        request_id_counter += 1

        # Create synthetic KV data
        num_heads = 8
        head_dim = 64
        num_blocks = int(avg_kv_size_mb / 0.5)  # ~0.5 MB per block

        kv_blocks = {
            j: np.random.randn(num_heads, head_dim).astype(np.float32)
            for j in range(num_blocks)
        }
        block_table = list(range(num_blocks))

        # Save context
        if swapper.save_context(request_id, kv_blocks, block_table, current_position=512):
            saved_count += 1

        # Restore some contexts
        if i % 3 == 0 and i > 0:
            result = swapper.restore_context(request_id)
            if result:
                restored_count += 1

    stats = swapper.stats()
    stats['saved_count'] = saved_count
    stats['restored_count'] = restored_count
    stats['num_requests'] = num_requests

    return stats
