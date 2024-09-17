#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

void showAVX512(const __m512i& v512) {
  alignas(alignof(__m512i)) unsigned char z[sizeof(__m512i)] = {0};
  _mm512_store_si512(reinterpret_cast<__m512i*>(z), v512);
  for (const auto& b : z) {
    std::cout << static_cast<unsigned int>(b) << ' ';
  }
  std::cout << "\n" << std::endl;
}

void test_avx2(){
    // AVX2
    __m256d a;
    __m256d b;
    __m256d c;
    _mm256_fnmadd_pd(a, b, c);
}

void test_avx512(){
    // AVX512
    const __m512i off0 = _mm512_set_epi64(3,0,2,0,1,0,0,0);
}

template <int Shift> void _mm512_aesenc_epi128_internal(const __m512i &v0, const __m512i &key, __m512i &res){
    __m128i v0_i = _mm512_extracti64x2_epi64(v0, Shift);
    __m128i key_i = _mm512_extracti64x2_epi64(key, Shift);
    __m128i res_aes_128 = _mm_aesenc_si128(v0_i, key_i);
    res = _mm512_inserti64x2(res, res_aes_128, Shift);
}

// test _mm512_aesenc_epi128
void test_vaes(){
    // VAES
    const __m512i v0 = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i key = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);
    __m512i res_aes_512 = _mm512_aesenc_epi128(v0, key);

    // AES
    __m512i res_aes_without_vaes;
    _mm512_aesenc_epi128_internal<0>(v0, key, res_aes_without_vaes);
    _mm512_aesenc_epi128_internal<1>(v0, key, res_aes_without_vaes);
    _mm512_aesenc_epi128_internal<2>(v0, key, res_aes_without_vaes);
    _mm512_aesenc_epi128_internal<3>(v0, key, res_aes_without_vaes);
    
    // Compare
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(res_aes_512, res_aes_without_vaes);
    if (cmp == 0xFF) {
        printf("AES result == VAES result\n");
    } else {
        printf("AES result != VAES result\n");
    }
}

// test _mm512_permutex2var_epi8
void test_avx512_vbmi(){
    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i b = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);   
    const __m512i idx = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);   
    __m512i res = _mm512_permutex2var_epi8(a, idx, b);

}

// test _mm512_mask_compressstoreu_epi16
void test_avx512_vbmi2(){
    void *res = malloc(8 * sizeof(uint64_t));
    void *res_without_vbmi = malloc(8 * sizeof(uint64_t));

    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);
    const __mmask32 k = 0x0F;
    _mm512_mask_compressstoreu_epi16(res, k, a);

}

int main(int argc, char **argv) {
    test_avx2();
    test_avx512();
    test_vaes();
    test_avx512_vbmi();
    test_avx512_vbmi2();
}
