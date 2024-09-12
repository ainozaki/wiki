#include <cuda.h>

#include "device.h"

__global__ void g_add1(int *a){
    d_add1(a);
}

int main(){
    int *d_a;
    const int size = 10;
    cudaMalloc(&d_a, sizeof(int) * size);
    g_add1<<<1, size>>>(d_a);
    cudaDeviceSynchronize();
    return 0;
}
