#include <immintrin.h>

const char* dgemm_desc = "My awesome dgemm.";

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

    const int block_size = 64;

    int i, j, k, ii, jj, kk;

    // loop over blocks
    for (kk = 0; kk < M; kk += block_size)
    {
        for (jj = 0; jj < M; jj += block_size)
        {
            for (ii = 0; ii < M; ii += block_size)
            {
                // loop in the blocks
                int i_end = (ii + block_size > M) ? M : ii + block_size;
                int j_end = (jj + block_size > M) ? M : jj + block_size;
                int k_end = (kk + block_size > M) ? M : kk + block_size;

                for (j = jj; j < j_end; ++j)
                {
                    for (i = ii; i < i_end; i += 4)
                    {
                        // handle edge case where less then 4 elements
                        int remaining = M - i;
                        if (remaining < 4)
                        {
                            for (int ir = i; ir < M; ++ir)
                            {
                                double cij = C[j*M + ir];
                                for (k = kk; k < k_end; ++k)
                                {
                                    cij += A[k*M + ir] * B[j*M + k];
                                }
                                C[j*M + ir] = cij;
                            }
                            break;
                        }

                        // load i:i+3 elements from C
                        __m256d c_vec = _mm256_loadu_pd(&C[j*M + i]);

                        for (k = kk; k < k_end; ++k)
                        {
                            // load elements from A and B
                            __m256d a_vec = _mm256_loadu_pd(&A[k*M + i]);

                            __m256d b_val = _mm256_set1_pd(B[j*M + k]);

                            c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
                        }
                        _mm256_storeu_pd(&C[j*M + i], c_vec);
                    }
                }
            }
        }
    }





    // // get highest multiple of 4 for vectorization
    // int M4 = M - (M % 4);

    // for (j = 0; j < M; ++j) {
    //     for (i = 0; i < M4; i += 4) {
    //         // load C[j*M + i:i+3] into avx register (4 rows)
    //         __m256d c_vec = _mm256_loadu_pd(&C[j*M + i]);

    //         for (k = 0; k < M; ++k) {
    //             // load four elements 
    //             __m256d a_vec = _mm256_loadu_pd(&A[k*M + i]);

    //             __m256d b_val = _mm256_set1_pd(B[j*M + k]);

    //             // multiply and sum
    //             c_vec = _mm256_fmadd_pd(a_vec, b_val, c_vec);
    //         }

    //         // store result back into C vector
    //         _mm256_storeu_pd(&C[j*M + i], c_vec);
    //     }

    //     for (i = M4; i < M; ++i)
    //     {
    //         double cij = C[j*M + i];
    //         for (k = 0; k < M; ++k)
    //         {
    //             cij += A[k*M + i] * B[j*M + k];
    //         }
    //         C[j*M + i] = cij;
    //     }
    // }
}
