/*
 * PyTorch C++ bindings for Context Swapping CUDA kernels
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" {
cudaError_t cu_swap_out(const float*, const int*, int, int, float*, float*, cudaStream_t);
cudaError_t cu_swap_in(const float*, const int*, int, int, float*, float*, cudaStream_t);
cudaError_t cu_zero_blocks(float*, const int*, int, int, cudaStream_t);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void swap_out(
    torch::Tensor kv_cache,      // [total_blocks, block_numel] CUDA
    torch::Tensor block_indices, // [num_blocks] CUDA int32
    torch::Tensor gpu_staging,   // [num_blocks, block_numel] CUDA
    torch::Tensor pinned_host    // [num_blocks, block_numel] pinned CPU
) {
    CHECK_INPUT(kv_cache);
    CHECK_INPUT(block_indices);
    CHECK_INPUT(gpu_staging);
    CHECK_CONTIGUOUS(pinned_host);
    TORCH_CHECK(pinned_host.is_pinned(), "pinned_host must be pinned memory");

    int num_blocks = block_indices.size(0);
    int block_numel = kv_cache.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_swap_out(
        kv_cache.data_ptr<float>(),
        block_indices.data_ptr<int>(),
        num_blocks, block_numel,
        gpu_staging.data_ptr<float>(),
        pinned_host.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "swap_out failed: ", cudaGetErrorString(err));
}

void swap_in(
    torch::Tensor pinned_host,   // [num_blocks, block_numel] pinned CPU
    torch::Tensor block_indices, // [num_blocks] CUDA int32
    torch::Tensor gpu_staging,   // [num_blocks, block_numel] CUDA
    torch::Tensor kv_cache       // [total_blocks, block_numel] CUDA
) {
    CHECK_CONTIGUOUS(pinned_host);
    TORCH_CHECK(pinned_host.is_pinned(), "pinned_host must be pinned memory");
    CHECK_INPUT(block_indices);
    CHECK_INPUT(gpu_staging);
    CHECK_INPUT(kv_cache);

    int num_blocks = block_indices.size(0);
    int block_numel = kv_cache.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_swap_in(
        pinned_host.data_ptr<float>(),
        block_indices.data_ptr<int>(),
        num_blocks, block_numel,
        gpu_staging.data_ptr<float>(),
        kv_cache.data_ptr<float>(),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "swap_in failed: ", cudaGetErrorString(err));
}

void zero_blocks(
    torch::Tensor kv_cache,
    torch::Tensor block_indices
) {
    CHECK_INPUT(kv_cache);
    CHECK_INPUT(block_indices);

    int num_blocks = block_indices.size(0);
    int block_numel = kv_cache.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cu_zero_blocks(
        kv_cache.data_ptr<float>(),
        block_indices.data_ptr<int>(),
        num_blocks, block_numel,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "zero_blocks failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swap_out", &swap_out, "Gather + async D2H KV block swap-out (CUDA)");
    m.def("swap_in", &swap_in, "Async H2D + scatter KV block swap-in (CUDA)");
    m.def("zero_blocks", &zero_blocks, "Zero evicted KV blocks (CUDA)");
}
