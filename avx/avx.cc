#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <bitset>
#include <cassert>
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

template <int Shift> void _mm512_aesenc_epi128_emulate(__m512i &res, const __m512i &v0, const __m512i &key){
    __m128i v0_i = _mm512_extracti64x2_epi64(v0, Shift);
    __m128i key_i = _mm512_extracti64x2_epi64(key, Shift);
    __m128i res_aes_128 = _mm_aesenc_si128(v0_i, key_i);
    res = _mm512_inserti64x2(res, res_aes_128, Shift);
}

void test_mm512_aesenc_epi128(){
    // VAES
    const __m512i v0 = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i key = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);
    __m512i res_aes_512 = _mm512_aesenc_epi128(v0, key);

    // AES
    __m512i res_aes_without_vaes;
    _mm512_aesenc_epi128_emulate<0>(res_aes_without_vaes, v0, key);
    _mm512_aesenc_epi128_emulate<1>(res_aes_without_vaes, v0, key);
    _mm512_aesenc_epi128_emulate<2>(res_aes_without_vaes, v0, key);
    _mm512_aesenc_epi128_emulate<3>(res_aes_without_vaes, v0, key);
    
    // Compare
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(res_aes_512, res_aes_without_vaes);
    if (cmp == 0xFF) {
        printf("Test __mm512_aesenc_epi128 ... OK\n");
    } else {
        printf("Test __mm512_aesenc_epi128 ... FAIL\n");
    }
}

__m512i _mm512_aesenclast_epi128_emulate(__m512i a, __m512i RoundKey){
    __m512i res = _mm512_setzero_epi32();
    __m128i a_i = _mm512_extracti64x2_epi64(a, 0);
    __m128i rk_i = _mm512_extracti64x2_epi64(RoundKey, 0);
    __m128i res_i = _mm_aesenclast_si128(a_i, rk_i);
    res = _mm512_inserti64x2(res, res_i, 0);

    a_i = _mm512_extracti64x2_epi64(a, 1);
    rk_i = _mm512_extracti64x2_epi64(RoundKey, 1);
    res_i = _mm_aesenclast_si128(a_i, rk_i);
    res = _mm512_inserti64x2(res, res_i, 1);

    a_i = _mm512_extracti64x2_epi64(a, 2);
    rk_i = _mm512_extracti64x2_epi64(RoundKey, 2);
    res_i = _mm_aesenclast_si128(a_i, rk_i);
    res = _mm512_inserti64x2(res, res_i, 2);

    a_i = _mm512_extracti64x2_epi64(a, 3);
    rk_i = _mm512_extracti64x2_epi64(RoundKey, 3);
    res_i = _mm_aesenclast_si128(a_i, rk_i);
    res = _mm512_inserti64x2(res, res_i, 3);
    return res;
}

void test_mm512_aesenclast_epi128(){
    // VAES
    const __m512i v0 = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i key = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);
    __m512i res = _mm512_aesenclast_epi128(v0, key); 

    __m512i res_emulated = _mm512_aesenclast_epi128_emulate(v0, key);

    // Compare
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(res, res_emulated);
    if (cmp == 0xFF) {
        printf("Test __mm512_aesenclast_epi128 ... OK\n");
    } else {
        printf("Test __mm512_aesenclast_epi128 ... FAIL\n");
    }
}

int extract_bit(const char *ptr, int offset_bit, int bits){
    assert(bits <= 32);
    assert(offset_bit % 8 == 0);
    uint8_t *ptr_i = (uint8_t *)ptr;
    return ptr_i[offset_bit / 8] & ((1 << bits) - 1);
}

__m512i _mm512_permutexvar_epi8_emulate(const __m512i &idx, const __m512i &a){
    alignas(alignof(__m512i)) unsigned char res[sizeof(__m512i)] = {0};
    const char *a_ptr = (char *)&a;
    const char *idx_ptr = (char *)&idx;
    for (int j = 0; j < 64; j++) {
        const int i = j * 8;
        const int offset = extract_bit(idx_ptr, i, 6);
        memcpy((char *)res + j, a_ptr + offset, 1);
    }
    return *(__m512i *)res;
}

void test_mm512_permutexvar_epi8(){
    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i idx = _mm512_set_epi64(0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF);
    __m512i res = _mm512_permutexvar_epi8(idx, a);

    __m512i res_emulated = _mm512_permutexvar_epi8_emulate(idx, a);

    // Compare
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(res, res_emulated);
    if (cmp == 0xFF) {
        printf("Test __mm512_permutexvar_epi8 ... OK\n");
    } else {
        printf("Test __mm512_permutexvar_epi8 ... FAIL\n");
    }  
}

__m512i _mm512_permutex2var_epi8_emulate(const __m512i &a, const __m512i &idx, const __m512i &b){
    alignas(alignof(__m512i)) unsigned char res[sizeof(__m512i)] = {0};
    const char *a_ptr = (char *)&a;
    const char *b_ptr = (char *)&b;
    const char *idx_ptr = (char *)&idx;
    for (int j = 0; j < 64; j++) {
        const int i = j * 8;
        const int offset = extract_bit(idx_ptr, i, 6);
        if (extract_bit(idx_ptr, i, 7) >> 6) {
            memcpy((char *)res + j, b_ptr + offset, 1);
        } else {
            memcpy((char *)res + j, a_ptr + offset, 1);
        }
    }
    return *(__m512i *)res;
}

void test_mm512_permutex2var_epi8(){
    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i b = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);   
    const __m512i idx = _mm512_set_epi64(0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF);
    __m512i res = _mm512_permutex2var_epi8(a, idx, b);

    __m512i res_emulated = _mm512_permutex2var_epi8_emulate(a, idx, b);

    // Compare
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(res, res_emulated);
    if (cmp == 0xFF) {
        printf("Test __mm512_permutex2var_epi8 ... OK\n");
    } else {
        printf("Test __mm512_permutex2var_epi8 ... FAIL\n");
        std::cout << "Input: " << std::endl;
        showAVX512(a);
        showAVX512(b);
        showAVX512(idx);
        std::cout << "HW: " << std::endl;
        showAVX512(res);
        std::cout << "Emulation: " << std::endl;
        showAVX512(res_emulated);
    }
}

void _mm512_mask_compressstoreu_epi16_emulate(void* base_addr, __mmask32 k, __m512i a){
    char *base = (char *)base_addr;
    for (int j = 0; j < 32; j++){
        const int i = j * 16;
        int k_j = (k >> j) & 1;
        if (k_j) {
            memcpy((char *)base, (char *)&a + i / 8, 2);
            base += 2;
        }
    }
}

void test_mm512_mask_compressstoreu_epi16(){
    alignas(alignof(__m512i)) unsigned char res[sizeof(__m512i)] = {0};
    alignas(alignof(__m512i)) unsigned char res_emulated[sizeof(__m512i)] = {0};
    
    const __m512i a = _mm512_set_epi64(0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF);
    const __mmask32 k = 15;
    _mm512_mask_compressstoreu_epi16(res, k, a);

    _mm512_mask_compressstoreu_epi16_emulate(res_emulated, k, a);
    
    // Compare
    if (memcmp(res, (char *)&res_emulated, sizeof(__m512i)) == 0) {
        printf("Test _mm512_mask_compressstoreu_epi16 ... OK\n");
    } else {
        std::cout << "Input: " << std::endl;
        showAVX512(*(__m512i *)&a);
        std::cout << "Mask: " << std::bitset<32>(k) << std::endl;
        std::cout << "HW: " << std::endl;
        showAVX512(*(__m512i *)res);
        std::cout << "Emulation: " << std::endl;
        showAVX512(*(__m512i *)res_emulated);
        printf("Test _mm512_mask_compressstoreu_epi16 ... FAIL\n");
    }
}

int main(int argc, char **argv) {
    test_avx2();
    test_avx512();
    test_mm512_aesenc_epi128();
    test_mm512_aesenclast_epi128();
    test_mm512_permutexvar_epi8();
    test_mm512_permutex2var_epi8();
    test_mm512_mask_compressstoreu_epi16();
}
