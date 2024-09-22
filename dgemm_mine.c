#include <immintrin.h>

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    int i, j, k;

    // get highest multiple of 4 for vectorization
    int M4 = M - (M % 4);

    for (j = 0; j < M; ++j) {
        for (i = 0; i < M4; i += 4) {
            // load C[j*M + i:i+3] into avx register (4 rows)
            __m256d c_vec = _mm256_loadu_pd(&C[j*M + i]);

            for (k = 0; k < M; ++k) {
                // load four elements 
                __m256d a_vec = _mm256_loadu_pd(&A[k*M + i]);

                __m256d b_val = _mm256_set1_pd(B[j*M + k]);

                // multiply and sum
                c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
            }

            // store result back into C vector
            _mm256_storeu_pd(&C[j*M + i], c_vec);
        }

        for (i = M4; i < M; ++i)
        {
            double cij = C[j*M + i];
            for (k = 0; k < M; ++k)
            {
                cij += A[k*M + i] * B[j*M + k];
            }
            C[j*M + i] = cij;
        }
    }
}
