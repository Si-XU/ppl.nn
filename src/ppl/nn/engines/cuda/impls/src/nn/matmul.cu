#include "cudakernel/matmul.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE 16

inline __host__ __device__ int iDivUp(const int& a, const int& b)
{
    int result = a % b != 0 ? (a < b ? 1 : a / b + 1) : a / b;
    return result;
}

//matrix_multiplication kernel 
__global__ void matmul_kernel(const float* d_a,const float* d_b, float* d_c, int M, int N, int K)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if( x > N - 1 || y > M -1)
        return;

    float dst = 0.0f;
    for(int i = 0; i < K; ++i)
    {
        dst += d_a[K * y + i] * d_b[i * N + x];
    }
    d_c[y * N + x] = dst;

}

void matmult_cuda(int M, int N, int K, const float* d_a, const float* d_b, float* d_c, const cudaStream_t& stream){
    const dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE);
    const dim3 gridDim(iDivUp(N,blockDim.x),iDivUp(M,blockDim.y));

    matmul_kernel<<<gridDim,blockDim,0,stream>>>(d_a, d_b, d_c, M, N, K);
    cudaGetLastError();
}

ppl::common::RetCode PPLCUDAMatmulForwardImp(
    const cudaStream_t& stream,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    const ppl::nn::TensorShape* weight_shape,
    const void* weight,
    const ppl::nn::TensorShape* output_shape,
    void* output) {

	auto dim_count = output_shape->GetDimCount();

    auto M = input_shape->GetDim(dim_count - 2);
	auto K = input_shape->GetDim(dim_count - 1);
	auto N = output_shape->GetDim(dim_count - 1);

	uint32_t loop_times = 1;
	for (uint32_t i = 0; i < dim_count - 2; i++) {
		loop_times *= output_shape->GetDim(i);
	}

	for (uint32_t i = 0; i < loop_times; i++) {
		auto input_offset = (const float*)input + (i * M * K);
		auto weight_offset = (const float*)weight + (i * K * N);
		auto output_offset = (float*)output + (i * M * N);

		if (weight_shape->GetDimCount() == 2) {
			weight_offset = (const float*)weight;
		}

		matmult_cuda(M, N, K, input_offset, weight_offset, output_offset, stream);
	}

	return ppl::common::RC_SUCCESS;

}