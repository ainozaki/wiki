#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define M 16
#define N 16
#define K 16

__global__ void tensor_core_gemm(half *a, half *b, float *c)
{
    // Shared memory to hold input tiles
    __shared__ half a_tile[M * K];
    __shared__ half b_tile[K * N];

    // Load tiles from global memory
    int tx = threadIdx.x;

    for (int i = tx; i < M * K; i += blockDim.x)
        a_tile[i] = a[i];
    for (int i = tx; i < K * N; i += blockDim.x)
        b_tile[i] = b[i];
    __syncthreads();

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;

    // Initialize output to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a_tile, K);
    wmma::load_matrix_sync(b_frag, b_tile, N);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    // Store the result
    wmma::store_matrix_sync(c, acc_frag, N, wmma::mem_row_major);
}

int main()
{
    half *a, *b;
    float *c;
    half *d_a, *d_b;
    float *d_c;

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(float);

    // Allocate and initialize host memory
    a = (half *)malloc(size_a);
    b = (half *)malloc(size_b);
    c = (float *)malloc(size_c);

    for (int i = 0; i < M * K; i++)
        a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++)
        b[i] = __float2half(1.0f);

    // Allocate device memory
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Copy data to device
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

    // Launch kernel
    tensor_core_gemm<<<1, 32>>>(d_a, d_b, d_c);

    // Copy result back
    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Print results
    printf("Result matrix C:\n");
    for (int i = 0; i < M * N; i++)
    {
        printf("%f ", c[i]);
        if ((i + 1) % N == 0)
            printf("\n");
    }

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}