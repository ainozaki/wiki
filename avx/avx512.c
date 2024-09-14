#include <immintrin.h>
#include <iostream>

int main() {
    // 512-bitのベクトルに16個のfloat値を格納
    alignas(64) float a[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                               9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    alignas(64) float b[16] = {16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0,
                               8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    alignas(64) float result[16];

    // AVX-512を使ってベクトルをロード
    __m512 vec_a = _mm512_load_ps(a);
    __m512 vec_b = _mm512_load_ps(b);

    // ベクトル加算
    __m512 vec_result = _mm512_add_ps(vec_a, vec_b);

    // 結果を保存
    _mm512_store_ps(result, vec_result);

    // 結果を表示
    for (int i = 0; i < 16; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

