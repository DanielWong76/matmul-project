#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

void micro_kernel(const double* A, const double* B, double* C, int i, int j, int K, int M)
{
    __m512d a_vec, b_val, c0, c1, c2, c3, c4, c5, c6, c7;

    // Load initial values from C
    c0 = _mm512_load_pd(&C[j * M + i]);
    c1 = _mm512_load_pd(&C[(j + 1) * M + i]);
    c2 = _mm512_load_pd(&C[(j + 2) * M + i]);
    c3 = _mm512_load_pd(&C[(j + 3) * M + i]);
    c4 = _mm512_load_pd(&C[(j + 4) * M + i]);
    c5 = _mm512_load_pd(&C[(j + 5) * M + i]);
    c6 = _mm512_load_pd(&C[(j + 6) * M + i]);
    c7 = _mm512_load_pd(&C[(j + 7) * M + i]);

    // Perform the multiplication and accumulate
    for (int k = 0; k < K; ++k) {
        a_vec = _mm512_load_pd(&A[i + k * M]); // Load 8 elements from A (row-wise)

        // Broadcast the corresponding elements from B
        b_val = _mm512_set1_pd(B[k + j * M]); // Broadcast B[k, j]
        c0 = _mm512_fmadd_pd(a_vec, b_val, c0); // c0 += A[i, k] * B[k, j]

        b_val = _mm512_set1_pd(B[k + (j + 1) * M]); // Broadcast B[k, j+1]
        c1 = _mm512_fmadd_pd(a_vec, b_val, c1);

        b_val = _mm512_set1_pd(B[k + (j + 2) * M]); // Broadcast B[k, j+2]
        c2 = _mm512_fmadd_pd(a_vec, b_val, c2);

        b_val = _mm512_set1_pd(B[k + (j + 3) * M]); // Broadcast B[k, j+3]
        c3 = _mm512_fmadd_pd(a_vec, b_val, c3);

        b_val = _mm512_set1_pd(B[k + (j + 4) * M]); // Broadcast B[k, j+4]
        c4 = _mm512_fmadd_pd(a_vec, b_val, c4);

        b_val = _mm512_set1_pd(B[k + (j + 5) * M]); // Broadcast B[k, j+5]
        c5 = _mm512_fmadd_pd(a_vec, b_val, c5);

        b_val = _mm512_set1_pd(B[k + (j + 6) * M]); // Broadcast B[k, j+6]
        c6 = _mm512_fmadd_pd(a_vec, b_val, c6);

        b_val = _mm512_set1_pd(B[k + (j + 7) * M]); // Broadcast B[k, j+7]
        c7 = _mm512_fmadd_pd(a_vec, b_val, c7);
    }

    // Store results back to C
    _mm512_store_pd(&C[j * M + i], c0);
    _mm512_store_pd(&C[(j + 1) * M + i], c1);
    _mm512_store_pd(&C[(j + 2) * M + i], c2);
    _mm512_store_pd(&C[(j + 3) * M + i], c3);
    _mm512_store_pd(&C[(j + 4) * M + i], c4);
    _mm512_store_pd(&C[(j + 5) * M + i], c5);
    _mm512_store_pd(&C[(j + 6) * M + i], c6);
    _mm512_store_pd(&C[(j + 7) * M + i], c7);
}


void avx512mult(const int block_size, const double *A_block, const double *B_block, double *C_block, int M) 
{
    int i, j;

    for (j = 0; j < block_size; j += 8)
    {
        for (i = 0; i < block_size; i += 8)
        {
            micro_kernel(A_block, B_block, C_block, i, j, block_size, block_size);
        }
    }
}

void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int block_size = 32;
    size_t alignment = 64;
    size_t block_bytes = block_size * block_size * sizeof(double);

    int i, j, k, ii, jj, kk;

    // Allocate memory for blocks (block_size x block_size), zero-initialized
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
                int i_end = (ii + block_size > M) ? (M - ii) : block_size;
                int j_end = (jj + block_size > M) ? (M - jj) : block_size;
                int k_end = (kk + block_size > M) ? (M - kk) : block_size;

                // Copy A_block
                memset(A_block, 0, block_bytes);
                for (k = 0; k < k_end; ++k) {
                    memcpy(&A_block[k * block_size], &A[(kk + k) * M + ii], i_end * sizeof(double));
                }

                // Copy B_block
                memset(B_block, 0, block_bytes);
                for (j = 0; j < j_end; ++j) {
                    memcpy(&B_block[j * block_size], &B[(jj + j) * M + kk], k_end * sizeof(double));
                }

                // Copy C_block
                memset(C_block, 0, block_bytes);
                for (j = 0; j < j_end; ++j) {
                    memcpy(&C_block[j * block_size], &C[(jj + j) * M + ii], i_end * sizeof(double));
                }

                avx512mult(block_size, A_block, B_block, C_block, M);

                // Copy result to C
                for (j = 0; j < j_end; ++j) {
                    memcpy(&C[(jj + j) * M + ii], &C_block[j * block_size], i_end * sizeof(double));
                }
            }
        }
    }

    free(A_block);
    free(B_block);
    free(C_block);
}
