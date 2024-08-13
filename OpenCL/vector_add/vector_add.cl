// kernel.cl
__kernel void
test(__global int* a, __global int* b, __global int* c, int n)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}