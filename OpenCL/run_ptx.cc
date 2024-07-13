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

    for (int i = 8; i < 24; i++){
        const int N = 1 << i;

        // Set up kernel execution parameters (example for a simple kernel)
        // Number of bytes to allocate for N ints
	    size_t bytes = N*sizeof(int);

        // Allocate memory for arrays A, B, and C on host
	    int *A = (int*)malloc(bytes);
	    int *B = (int*)malloc(bytes);
	    int *C = (int*)malloc(bytes);

        // Allocate memory for arrays d_A, d_B, and d_C on device
        int *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // Fill host arrays A and B
        for(int i=0; i<N; i++)
        {
            A[i] = 1.0;
            B[i] = 2.0;
        }

        // Copy data from host arrays A and B to device arrays d_A and d_B
        cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

        // Set execution configuration parameters
        //		thr_per_blk: number of CUDA threads per grid block
        //		blk_in_grid: number of blocks in grid
        int thr_per_blk = 8;
        int blk_in_grid = N / thr_per_blk;
        char *args[] = { (char*)&d_A, (char*)&d_B, (char*)&d_C, (char*)&N };
        

        // 10 iterations
        double total_time = 0.0;
        for (int j = 0; j < 10; j++) {
            // Timer
            struct timeval start_time, end_time;
            gettimeofday(&start_time, NULL);
            // Launch kernel
            checkCudaErrors(cuLaunchKernel(cuFunction,  // function to launch
                                        blk_in_grid, 1, 1,    // grid dim
                                        thr_per_blk, 1, 1,   // block dim
                                        0,                 // shared memory
                                        NULL,              // stream
                                            (void **)args, 
                                            NULL));       // arguments
            checkCudaErrors(cuCtxSynchronize());
            gettimeofday(&end_time, NULL);
            double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
            total_time += elapsed_time;
        }
        std::cout << "N: 2 ** " << i << ", " << total_time * 100 << " ms" << std::endl;

        // Copy data from device array d_C to host array C
        cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

        // Verify results
        int count = N;
        int tolerance = 1.0e-14;
        for(int i=0; i<N; i++)
        {
            if( std::fabs(C[i] - 3.0) > tolerance)
            { 
                printf("Error: value of C[%d] = %d instead of 3.0\n", i, C[i]);
                count -= 1;
            }
        }	
        if (count != N){
            std::cout << "Failed: " << count << "/" << N << " are correct." << std::endl;
        }

        // Free
        free(A);
        free(B);
        free(C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }    
    return 0;
}
