#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

// test _mm512_aesenc_epi128
void test_vaes(){
    // VAES
    const __m512i v0 = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i key = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);
    __m512i res_aes_512 = _mm512_aesenc_epi128(v0, key);

    // AES
    __m512i res_aes_without_vaes;
    __m128i v0_i = _mm512_extracti64x2_epi64(v0, 0);
    __m128i key_i = _mm512_extracti64x2_epi64(key, 0);
    __m128i res_aes_128 = _mm_aesenc_si128(v0_i, key_i);
    res_aes_without_vaes = _mm512_inserti64x2(res_aes_without_vaes, res_aes_128, 0);

    v0_i = _mm512_extracti64x2_epi64(v0, 1);
    key_i = _mm512_extracti64x2_epi64(key, 1);
    res_aes_128 = _mm_aesenc_si128(v0_i, key_i);
    res_aes_without_vaes = _mm512_inserti64x2(res_aes_without_vaes, res_aes_128, 1);

    v0_i = _mm512_extracti64x2_epi64(v0, 2);
    key_i = _mm512_extracti64x2_epi64(key, 2);
    res_aes_128 = _mm_aesenc_si128(v0_i, key_i);
    res_aes_without_vaes = _mm512_inserti64x2(res_aes_without_vaes, res_aes_128, 2);

    v0_i = _mm512_extracti64x2_epi64(v0, 3);
    key_i = _mm512_extracti64x2_epi64(key, 3);
    res_aes_128 = _mm_aesenc_si128(v0_i, key_i);
    res_aes_without_vaes = _mm512_inserti64x2(res_aes_without_vaes, res_aes_128, 3);

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

    __m512i res_without_vbmi;
    __m128i a_i = _mm512_extracti64x2_epi64(a, 0);
    __m128i b_i = _mm512_extracti64x2_epi64(b, 0);
    __m128i idx_i = _mm512_extracti64x2_epi64(idx, 0);
    __m128i res_i = _mm_permutex2var_epi8(a_i, idx_i, b_i);
    res_without_vbmi = _mm512_inserti64x2(res_without_vbmi, res_i, 0);

    a_i = _mm512_extracti64x2_epi64(a, 1);
    b_i = _mm512_extracti64x2_epi64(b, 1);
    idx_i = _mm512_extracti64x2_epi64(idx, 1);
    res_i = _mm_permutex2var_epi8(a_i, idx_i, b_i);
    res_without_vbmi = _mm512_inserti64x2(res_without_vbmi, res_i, 1);

    a_i = _mm512_extracti64x2_epi64(a, 2);
    b_i = _mm512_extracti64x2_epi64(b, 2);
    idx_i = _mm512_extracti64x2_epi64(idx, 2);
    res_i = _mm_permutex2var_epi8(a_i, idx_i, b_i);
    res_without_vbmi = _mm512_inserti64x2(res_without_vbmi, res_i, 2);

    a_i = _mm512_extracti64x2_epi64(a, 3);
    b_i = _mm512_extracti64x2_epi64(b, 3);
    idx_i = _mm512_extracti64x2_epi64(idx, 3);
    res_i = _mm_permutex2var_epi8(a_i, idx_i, b_i);
    res_without_vbmi = _mm512_inserti64x2(res_without_vbmi, res_i, 3);

    // Compare
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(res, res_without_vbmi);
    if (cmp == 0xFF) {
        printf("VBMI result == without VBMI result\n");
    } else {
        printf("VBMI result != without VBMI result\n");
    }
}

// test _mm512_mask_compressstoreu_epi16
void test_avx512_vbmi2(){
    void *res = malloc(8 * sizeof(uint64_t));
    void *res_without_vbmi = malloc(8 * sizeof(uint64_t));

    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);
    const __mmask32 k = 0x0F;
    _mm512_mask_compressstoreu_epi16(res, k, a);

    // Without VBMI2
    __m128i a_i = _mm512_extracti64x2_epi64(a, 0);
    __mmask8 k_i = k & 0xFF;
    _mm_mask_compressstoreu_epi16(res_without_vbmi, k_i, a_i);

    a_i = _mm512_extracti64x2_epi64(a, 1);
    k_i = (k >> 8) & 0xFF;
    _mm_mask_compressstoreu_epi16(res_without_vbmi + 128, k_i, a_i);

    a_i = _mm512_extracti64x2_epi64(a, 2);
    k_i = (k >> 16) & 0xFF;
    _mm_mask_compressstoreu_epi16(res_without_vbmi + 256, k_i, a_i);

    a_i = _mm512_extracti64x2_epi64(a, 3);
    k_i = (k >> 24) & 0xFF;
    _mm_mask_compressstoreu_epi16(res_without_vbmi + 384, k_i, a_i);

    // Compare
    int cmp = memcmp(res, res_without_vbmi, 8 * sizeof(uint64_t));
    if (cmp == 0) {
        printf("VBMI2 result == without VBMI2 result\n");
    } else {
        printf("VBMI2 result != without VBMI2 result\n");
    }
}

int main(int argc, char **argv) {
    test_avx2();
    test_avx512();
    test_vaes();
    test_avx512_vbmi();
    test_avx512_vbmi2();
}
