#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "max shared memory (per block): "
              << prop.sharedMemPerBlock / 1000 << " KB" << std::endl;
    std::cout << "max shared memory (per SM): "
              << prop.sharedMemPerMultiprocessor / 1000 << " KB" << std::endl;

    return 0;
}

