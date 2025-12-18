# GPUでのGEMM最適化

[行列乗算の最適化入門（GPGPU編）](https://lpha-z.hatenablog.com/entry/2024/07/14/231500)
```
__global__ void kernel( double* a, double* b, double* c ) {
    int i0 = blockIdx.y * SizeI;
    int j0 = blockIdx.x * SizeJ;
    int i2 = threadIdx.y;
    int j2 = threadIdx.x;

    double sum[RegisterBlockJ][RegisterBlockI];
    for( int i1 = 0; i1 < RegisterBlockI; ++i1 )
    for( int j1 = 0; j1 < RegisterBlockJ; ++j1 )
    {
        int i = i0 + (i1 * ThreadsPerBlockI) + i2;
        int j = j0 + (j1 * ThreadsPerBlockJ) + j2;
        sum[j1][i1] = c[i*N+j];
    }

    for( int k = 0; k < N; ++k )
    for( int i1 = 0; i1 < RegisterBlockI; ++i1 )
    for( int j1 = 0; j1 < RegisterBlockJ; ++j1 )
    {
        int i = i0 + (i1 * ThreadsPerBlockI) + i2;
        int j = j0 + (j1 * ThreadsPerBlockJ) + j2;
        sum[j1][i1] = fma( a[i*N+k], b[k*N+j], sum[j1][i1] );
    }

    for( int i1 = 0; i1 < RegisterBlockI; ++i1 )
    for( int j1 = 0; j1 < RegisterBlockJ; ++j1 )
    {
        int i = i0 + (i1 * ThreadsPerBlockI) + i2;
        int j = j0 + (j1 * ThreadsPerBlockJ) + j2;
        c[i*N+j] = sum[j1][i1];
    }
}
```
- k軸はTime, J軸はBlock, 
 - (N, K, M) = (4096, 4096, 4096) の行列積
 - Thread あたり 4 x 16 Register
 - Block あたり 128 x 1 Thread
 - = Block あたり 16 x 512 要素を計算
 - 各 Thread がどこを計算するか？
   - ブロックの左上の要素：(i0, j0) = (blockIdx.y * 16, blockIdx.x * 512)
   - 各スレッドのループ回数：(i1, j1) = (16, 4)
   - スレッド内のオフセット：(i1 * 16 + threadIdx.y, j1 * 1 + threadIdx.x)
    - 隣あうThreadIdx.xが連続アクセスするようになる 
- sumをレジスタで保持する
  - GPUはL1はwrite-through、L2はwrite-back cache であり、レジスタを使わないとなるとL2まで書き込みに行ってしまうため
- 同じ cache line (128bit) から読み出せば早い、連続 cache line ならそれなりに早い
- L1 cache はコンパイラが Read-only だと判断したデータがキャッシュされる
