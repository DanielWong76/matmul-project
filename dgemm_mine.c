#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

void avx512mult(const int block_size, const double *A_block, const double *B_block, double *C_block) 
{
    int i, j, k;

    for (j = 0; j < block_size; ++j)
    {
        for (i = 0; i < block_size; i += 8)
        {
            // AVX-512 code for 8 elements
            __m512d c_vec = _mm512_load_pd(&C_block[j*block_size + i]);
            for (k = 0; k < block_size; ++k)
            {
                __m512d a_vec = _mm512_load_pd(&A_block[k*block_size + i]);
                __m512d b_val = _mm512_set1_pd(B_block[j*block_size + k]);
                c_vec = _mm512_fmadd_pd(a_vec, b_val, c_vec);
            }
            _mm512_store_pd(&C_block[j*block_size + i], c_vec);
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{

    const int block_size = 32;
    size_t alignment = 64;
    size_t block_bytes = block_size * block_size * sizeof(double);

    int i, j, k, ii, jj, kk;

    // Allocate memory for blocks (block_size x block_size)
    double* A_block = aligned_alloc(alignment, block_bytes);
    double* B_block = aligned_alloc(alignment, block_bytes);
    double* C_block = aligned_alloc(alignment, block_bytes);

    // loop over blocks
    for (kk = 0; kk < M; kk += block_size)
    {
        for (jj = 0; jj < M; jj += block_size)
        {
            for (ii = 0; ii < M; ii += block_size)
            {
                // loop in the blocks
                int i_end = (ii + block_size > M) ? (M - ii) : block_size;
                int j_end = (jj + block_size > M) ? (M - jj) : block_size;
                int k_end = (kk + block_size > M) ? (M - kk) : + block_size;

                // copy blocks into temporary buffers (so it is aligned in memory)
                for (k = 0; k < k_end; ++k) {
                    memcpy(&A_block[k * block_size], &A[(kk + k) * M + ii], i_end * sizeof(double));
                        if (i_end < block_size)
                            memset(&A_block[k * block_size + i_end], 0, (block_size - i_end) * sizeof(double));
                }

                // zero pad if necessary
                for (k = k_end; k < block_size; ++k)
                    memset(&A_block[k * block_size], 0, block_size * sizeof(double));

                // Copy B_block
                for (j = 0; j < j_end; ++j)
                {
                    memcpy(&B_block[j * block_size], &B[(jj + j) * M + kk], k_end * sizeof(double));
                    if (k_end < block_size)
                        memset(&B_block[j * block_size + k_end], 0, (block_size - k_end) * sizeof(double));
                }
                // Zero-pad remaining rows if necessary
                for (j = j_end; j < block_size; ++j)
                    memset(&B_block[j * block_size], 0, block_size * sizeof(double));

                // Copy C_block
                for (j = 0; j < j_end; ++j)
                {
                    memcpy(&C_block[j * block_size], &C[(jj + j) * M + ii], i_end * sizeof(double));
                    if (i_end < block_size)
                        memset(&C_block[j * block_size + i_end], 0, (block_size - i_end) * sizeof(double));
                }
                // Zero-pad remaining rows if necessary
                for (j = j_end; j < block_size; ++j)
                    memset(&C_block[j * block_size], 0, block_size * sizeof(double));

                avx512mult(block_size, A_block, B_block, C_block);

                // copy result to C again
                for (j = 0; j < j_end; ++j)
                {
                    memcpy(&C[(jj + j) * M + ii], &C_block[j * block_size], i_end * sizeof(double));
                }

                // for (j = jj; j < j_end; ++j)
                // {
                //     for (i = ii; i < i_end; i += 8)
                //     {
                //         int remaining = M - i;
                //         if (remaining < 8)
                //         {
                //             if (remaining >= 4) {
                //                 // Use AVX-256 for 4 remaining elements
                //                 __m256d c_vec256 = _mm256_loadu_pd(&C[j*M + i]);
                //                 for (k = kk; k < k_end; ++k) {
                //                     __m256d a_vec256 = _mm256_loadu_pd(&A[k*M + i]);
                //                     __m256d b_val256 = _mm256_set1_pd(B[j*M + k]);
                //                     c_vec256 = _mm256_fmadd_pd(a_vec256, b_val256, c_vec256);
                //                 }
                //                 _mm256_storeu_pd(&C[j*M + i], c_vec256);
                //             }
                //             // Handle remaining scalar elements...
                //             for (int ir = i + (remaining >= 4 ? 4 : 0); ir < M; ++ir)
                //             {
                //                 double cij = C[j*M + ir];
                //                 for (k = kk; k < k_end; ++k)
                //                 {
                //                     cij += A[k*M + ir] * B[j*M + k];
                //                 }
                //                 C[j*M + ir] = cij;
                //             }
                //             break;
                //         }
                //         // AVX-512 code for 8 elements
                //         __m512d c_vec = _mm512_loadu_pd(&C[j*M + i]);
                //         for (k = kk; k < k_end; ++k)
                //         {
                //             __m512d a_vec = _mm512_loadu_pd(&A[k*M + i]);
                //             __m512d b_val = _mm512_set1_pd(B[j*M + k]);
                //             c_vec = _mm512_fmadd_pd(a_vec, b_val, c_vec);
                //         }
                //         _mm512_storeu_pd(&C[j*M + i], c_vec);
                //     }
                // }
            }
        }
    }

    free(A_block);
    free(B_block);
    free(C_block);



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
