#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <bitset>
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

// test _mm512_aesenc_epi128
void test_vaes(){
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
        printf("AES result == VAES result\n");
    } else {
        printf("AES result != VAES result\n");
    }
}

__m512i _mm512_permutex2var_epi8_internal(const __m512i &a, const __m512i &idx, const __m512i &b){
    __m512i res;
    for (int j = 0; j < 64; j++) {
        const int i = j * 8;
        // extract idx[i:i+5]
    }
    return res;
}

// test _mm512_permutex2var_epi8
void test_avx512_vbmi(){
    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);   
    const __m512i b = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);   
    const __m512i idx = _mm512_set_epi64(0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);   
    __m512i res = _mm512_permutex2var_epi8(a, idx, b);

}

void _mm512_mask_compressstoreu_epi16_emulate(void *res, const __m512i &v0, const __mmask32 &k){
    char *base = (char *)res;
    for (int i = 0; i < 32; i++){
        const int j = i * 16;
        int k_i = (k >> i) & 1;
        if (k_i) {
            memcpy((char *)base, (char *)&v0 + j, 2);
            base += 2;
        }
    }
}

// test _mm512_mask_compressstoreu_epi16
void test_avx512_vbmi2(){
    alignas(alignof(__m512i)) unsigned char res[sizeof(__m512i)] = {0};
    alignas(alignof(__m512i)) unsigned char res_without_vbmi[sizeof(__m512i)] = {0};
    
    const __m512i a = _mm512_set_epi64(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);
    const __mmask32 k = 16;
    _mm512_mask_compressstoreu_epi16(res, k, a);

    _mm512_mask_compressstoreu_epi16_emulate(res_without_vbmi, a, k);
    
    // Compare
    if (memcmp(res, (char *)&res_without_vbmi, sizeof(__m512i)) == 0) {
        printf("Compress result == Compress without VBMI result\n");
    } else {
        std::cout << "Input: " << std::endl;
        showAVX512(*(__m512i *)&a);
        std::cout << "Mask: " << std::bitset<32>(k) << std::endl;
        std::cout << "HW: " << std::endl;
        showAVX512(*(__m512i *)res);
        std::cout << "Emulation: " << std::endl;
        showAVX512(*(__m512i *)res_without_vbmi);
        printf("Compress result != Compress without VBMI result\n");
    }
}

int main(int argc, char **argv) {
    test_avx2();
    test_avx512();
    test_vaes();
    //test_avx512_vbmi();
    test_avx512_vbmi2();
}
