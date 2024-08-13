#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err, msg) if (err != cudaSuccess) { printf("%s failed: %s\n", msg, cudaGetErrorString(err)); exit(1); }

int power(int a, int n) {
    if (n == 1) return a;
    else if (n == 0) return 1;
    else if (n % 2 == 0) {
        int k = power(a, n / 2);
        return (k * k);
    } else if (n % 2 == 1) {
        int k = power(a, (n - 1) / 2);
        return (k * k * a);
    }
    return 0;
}

extern "C" __global__ void test(
    const int N,
    const int pN,
    const int a,
    const int b,
    const int g,
    const int m,
    const int p,
    int *x,
    int *x_copy,
    int *rou,
    const int kk
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (N >> 1)) return;

    int j2 = j << 1;
    int j_mod_a = j % a;
    int c, d;

    if (kk == 0) {
        c = x_copy[j];
        d = rou[(j_mod_a * b * pN) % (p - 1)] * x_copy[j + (N >> 1)];
        d = (d - ((d * m) >> 52) * p);
        if (d >= p) d -= p;
        if (d < 0) d += p;
        x[j2 - j_mod_a] = (c + d) % p;
        if (x[j2 - j_mod_a] < 0) x[j2 - j_mod_a] += p;
        x[j2 - j_mod_a + a] = (c - d) % p;
        if (x[j2 - j_mod_a + a] < 0) x[j2 - j_mod_a + a] += p;
    } else {
        c = x[j];
        d = rou[(j_mod_a * b * pN) % (p - 1)] * x[j + (N >> 1)];
        d = (d - ((d * m) >> 52) * p);
        if (d >= p) d -= p;
        if (d < 0) d += p;
        x_copy[j2 - j_mod_a] = (c + d) % p;
        if (x_copy[j2 - j_mod_a] < 0) x_copy[j2 - j_mod_a] += p;
        x_copy[j2 - j_mod_a + a] = (c - d) % p;
        if (x_copy[j2 - j_mod_a + a] < 0) x_copy[j2 - j_mod_a + a] += p;
    }
}

int main(int argc, char** argv) {
    int g = 17;
    int p = 3329;

    for (int n = 4; n <= 23; n++) {
        double total_time = 0.0;
        int N = 1 << n;
        for (int iii = 0; iii < 10; iii++) {
            cudaError_t err;

            int *x, *x_copy, *rou, *x_cpu;
            int *d_x, *d_x_copy, *d_rou;

            x = (int *)malloc(N * sizeof(int));
            x_copy = (int *)malloc(N * sizeof(int));
            rou = (int *)malloc((p - 1) * sizeof(int));
            x_cpu = (int *)malloc(N * sizeof(int));

            if (x == NULL || x_copy == NULL || rou == NULL || x_cpu == NULL) {
                printf("Error allocating memory\n");
                exit(1);
            }

            for (int i = 0; i < N; i++) {
                x[i] = 0;
                x_copy[i] = rand() % p;
                x_cpu[i] = x_copy[i];
            }

            rou[0] = 1;
            for (int i = 0; i < p - 2; i++) rou[i + 1] = (rou[i] * g) % p;


            err = cudaMalloc((void **)&d_x, N * sizeof(int));
            CHECK_CUDA_ERROR(err, "cudaMalloc d_x");
            err = cudaMalloc((void **)&d_x_copy, N * sizeof(int));
            CHECK_CUDA_ERROR(err, "cudaMalloc d_x_copy");
            err = cudaMalloc((void **)&d_rou, (p - 1) * sizeof(int));
            CHECK_CUDA_ERROR(err, "cudaMalloc d_rou");

            err = cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err, "cudaMemcpy d_x");
            err = cudaMemcpy(d_x_copy, x_copy, N * sizeof(int), cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err, "cudaMemcpy d_x_copy");
            err = cudaMemcpy(d_rou, rou, (p - 1) * sizeof(int), cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err, "cudaMemcpy d_rou");

            int pN = (p - 1) / N;
            int a = 1;
            int b = power(2, n - 1);
            double inv_p = 1.0 / (double)p;
            int m = (int)(inv_p * (double)((1ULL << 52) + 0.5));

            int kk = 0;
            size_t M = N >> 1;
            size_t threads_per_block = 256;
            size_t num_blocks = (M + threads_per_block - 1) / threads_per_block;

            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);
            for (int i = 0; i < n; i++) {
                test<<<num_blocks, threads_per_block>>>(N, pN, a, b, g, m, p, d_x, d_x_copy, d_rou, kk);
                err = cudaGetLastError();
                CHECK_CUDA_ERROR(err, "cudaGetLastError");
                err = cudaDeviceSynchronize();
                CHECK_CUDA_ERROR(err, "cudaDeviceSynchronize");

                a <<= 1;
                b >>= 1;
                kk = 1 - kk;
            }
            gettimeofday(&end_time, NULL);

            int *ans;
            if (kk == 1) {
                err = cudaMemcpy(x, d_x, N * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_CUDA_ERROR(err, "cudaMemcpy d_x");
                ans = x;
            } else {
                err = cudaMemcpy(x_copy, d_x_copy, N * sizeof(int), cudaMemcpyDeviceToHost);
                CHECK_CUDA_ERROR(err, "cudaMemcpy d_x_copy");
                ans = x_copy;
            }

            //printf("index 0: GPU = %d\n", ans[0]);

            cudaFree(d_x_copy);
            cudaFree(d_x);
            cudaFree(d_rou);

            free(x);
            free(x_copy);
            free(rou);
            free(x_cpu);

            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;
        }
        printf("Elapsed time for N = 2 ** %d: %f ms\n", n, total_time * 1000 / 10);
    }
    return 0;
}
