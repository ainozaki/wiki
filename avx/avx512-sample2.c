#include <immintrin.h>

int main(int argc, char **argv) {
    __m256d a;
    __m256d b;
    __m256d c;

    _mm256_fnmadd_pd(a, b, c);
}
