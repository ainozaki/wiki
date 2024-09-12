#include "device.h"

__device__ void d_add1(int *a){
	int idx = threadIdx.x;
	a[idx] = a[idx] + 1;
}
