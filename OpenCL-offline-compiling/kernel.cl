// kernel.cl
__kernel void
vecAdd(__global int* z, __global const int* x, __global const int* y, int n)
{
  for (int i = 0; i < n; i++) {
    printf("z[%d] = x[%d] + y[%d]\n", i, i, i);
    z[i] = x[i] + y[i];
  }
}