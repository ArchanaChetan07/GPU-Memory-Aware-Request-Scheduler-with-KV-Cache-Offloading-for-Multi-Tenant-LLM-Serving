/*
 * Request-Level Context Swapping: CUDA Kernels
 *
 * Implements:
 * - Batched KV block gather (GPU -> staging buffer) for swap-out
 * - Batched KV block scatter (staging buffer -> GPU) for swap-in
 * - Async pipelined transfers via dedicated streams
 *
 * The staging buffer is pinned host memory; cudaMemcpyAsync overlaps
 * with compute on the default stream.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_THREADS 256

/*
 * gather_kv_blocks_kernel
 *
 * Gathers scattered KV blocks into a contiguous staging buffer.
 * Grid: (num_blocks), Block: (256 threads)
 *
 * kv_cache layout: [total_blocks, block_numel]
 * staging layout:  [num_blocks, block_numel] (contiguous)
 */
__global__ void gather_kv_blocks_kernel(
    const float* kv_cache,
    const int* block_indices,   // [num_blocks] which cache blocks to gather
    int num_blocks,
    int block_numel,            // elements per block
    float* staging
) {
    int blk = blockIdx.x;
    if (blk >= num_blocks) return;

    int src_block = block_indices[blk];
    const float* src = kv_cache + (size_t)src_block * block_numel;
    float* dst = staging + (size_t)blk * block_numel;

    for (int i = threadIdx.x; i < block_numel; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/*
 * scatter_kv_blocks_kernel
 *
 * Scatters contiguous staging data back into KV cache blocks.
 * Inverse of gather.
 */
__global__ void scatter_kv_blocks_kernel(
    const float* staging,
    const int* block_indices,   // [num_blocks] destination cache blocks
    int num_blocks,
    int block_numel,
    float* kv_cache
) {
    int blk = blockIdx.x;
    if (blk >= num_blocks) return;

    int dst_block = block_indices[blk];
    const float* src = staging + (size_t)blk * block_numel;
    float* dst = kv_cache + (size_t)dst_block * block_numel;

    for (int i = threadIdx.x; i < block_numel; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/*
 * zero_kv_blocks_kernel
 *
 * Zeroes evicted blocks (optional hygiene, prevents stale reads).
 */
__global__ void zero_kv_blocks_kernel(
    float* kv_cache,
    const int* block_indices,
    int num_blocks,
    int block_numel
) {
    int blk = blockIdx.x;
    if (blk >= num_blocks) return;

    int target = block_indices[blk];
    float* dst = kv_cache + (size_t)target * block_numel;

    for (int i = threadIdx.x; i < block_numel; i += blockDim.x) {
        dst[i] = 0.0f;
    }
}

extern "C" {

/*
 * cu_swap_out: gather blocks to GPU staging, then async copy to pinned host.
 * The caller provides a GPU staging buffer and a pinned host buffer.
 */
cudaError_t cu_swap_out(
    const float* kv_cache,
    const int* block_indices,
    int num_blocks,
    int block_numel,
    float* gpu_staging,
    float* pinned_host,
    cudaStream_t stream
) {
    gather_kv_blocks_kernel<<<num_blocks, BLOCK_THREADS, 0, stream>>>(
        kv_cache, block_indices, num_blocks, block_numel, gpu_staging
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    size_t bytes = (size_t)num_blocks * block_numel * sizeof(float);
    return cudaMemcpyAsync(pinned_host, gpu_staging, bytes,
                           cudaMemcpyDeviceToHost, stream);
}

/*
 * cu_swap_in: async copy from pinned host to GPU staging, then scatter.
 */
cudaError_t cu_swap_in(
    const float* pinned_host,
    const int* block_indices,
    int num_blocks,
    int block_numel,
    float* gpu_staging,
    float* kv_cache,
    cudaStream_t stream
) {
    size_t bytes = (size_t)num_blocks * block_numel * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(gpu_staging, pinned_host, bytes,
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;

    scatter_kv_blocks_kernel<<<num_blocks, BLOCK_THREADS, 0, stream>>>(
        gpu_staging, block_indices, num_blocks, block_numel, kv_cache
    );
    return cudaGetLastError();
}

cudaError_t cu_zero_blocks(
    float* kv_cache,
    const int* block_indices,
    int num_blocks,
    int block_numel,
    cudaStream_t stream
) {
    zero_kv_blocks_kernel<<<num_blocks, BLOCK_THREADS, 0, stream>>>(
        kv_cache, block_indices, num_blocks, block_numel
    );
    return cudaGetLastError();
}

} // extern "C"
