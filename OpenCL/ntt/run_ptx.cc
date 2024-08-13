#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <sys/time.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

void check(CUresult err, const char *const func, const char *const file, const int line) {
    if (err != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(err, &errStr);
        std::cerr << "CUDA error = " << static_cast<unsigned int>(err) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        std::cerr << "Error string: " << errStr << std::endl;
        exit(1);
    }
}

std::string loadPTXFile(const char* filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main(int argc, char* argv[]) {
    int g = 17;
    int p = 3329;

    // Receive input PTX filename and Kernel name
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <PTX file> <Kernel name>" << std::endl;
        exit(1);
    }
    const char* ptxFilename = argv[1];
    const char* kernelName = argv[2];

    CUdevice cuDevice;
    CUcontext cuContext;
    CUmodule cuModule;
    CUfunction cuFunction;

    // Init CUDA Driver API
    checkCudaErrors(cuInit(0));

    // Get GPU device
    checkCudaErrors(cuDeviceGet(&cuDevice, 0));

    // Create CUDA context
    checkCudaErrors(cuCtxCreate(&cuContext, 0, cuDevice));

    // Load PTX file
    std::string ptxSource = loadPTXFile(ptxFilename);
    
    // Create CUDA module from PTX code
    checkCudaErrors(cuModuleLoadDataEx(&cuModule, ptxSource.c_str(), 0, 0, 0));

    // Get kernel function from the module
    checkCudaErrors(cuModuleGetFunction(&cuFunction, cuModule, kernelName));

    for (int i = 10; i < 24; i++){
        const int N = 1 << i;

        // Allocate memory for arrays A, B, and C on host
        int *d_x, *d_x_copy, *d_rou;
        int *x = (int *)malloc(N * sizeof(int));
        int *x_copy = (int *)malloc(N * sizeof(int));
        int *rou = (int *)malloc((p - 1) * sizeof(int));
        int *x_cpu = (int *)malloc(N * sizeof(int));

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
        for (int i = 0; i < p - 2; i++) {
            rou[i + 1] = (rou[i] * g) % p;
        }

        cudaMalloc((void **)&d_x, N * sizeof(int));
        cudaMalloc((void **)&d_x_copy, N * sizeof(int));
        cudaMalloc((void **)&d_rou, (p - 1) * sizeof(int));
        cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_copy, x_copy, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rou, rou, (p - 1) * sizeof(int), cudaMemcpyHostToDevice);
	    
        int pN = (p - 1) / N;
        int a = 1;
        int b = std::pow(2, i - 1);
        double inv_p = 1.0 / (double)p;
        int m = (int)(inv_p * (double)((1ULL << 52) + 0.5));

        int kk = 0;
        size_t M = N >> 1;
        size_t threads_per_block = 256;
        size_t num_blocks = (M + threads_per_block - 1) / threads_per_block;


        // 10 iterations
        double total_time = 0.0;
        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);
        for (int logi = 0; logi < i; logi++) {
            // Launch kernel
            char *args[] = {(char *)&N, (char *)&pN, (char *)&a, (char *)&b, (char *)&g, (char *)&m, (char *)&p, (char *)&d_x, (char *)&d_x_copy, (char *)&d_rou, (char *)&kk};
            checkCudaErrors(cuLaunchKernel(cuFunction,  // function to launch
                                        num_blocks, 1, 1,    // grid dim
                                        threads_per_block, 1, 1,   // block dim
                                        0,                 // shared memory
                                        NULL,              // stream
                                            (void **)args, 
                                            NULL));       // arguments
            checkCudaErrors(cuCtxSynchronize());
            a <<= 1;
            b >>= 1;
            kk = 1 - kk;
        }
        gettimeofday(&end_time, NULL);
        double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
        total_time += elapsed_time;
        std::cout << "N: 2 ** " << i << ", " << total_time * 100 << " ms" << std::endl;

        // Copy data from device array d_C to host array C

        // Free
        free(x);
        free(x_copy);
        free(rou);
        free(x_cpu);
        cudaFree(d_x);
        cudaFree(d_x_copy);
        cudaFree(d_rou);
    }    
    return 0;
}
