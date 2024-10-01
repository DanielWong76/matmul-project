#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <xmmintrin.h>

const char* dgemm_desc = "My awesome dgemm.";

#define min(x, y) ((x) < (y) ? (x) : (y))

// 16x6 matrix multiply kernel
void micro_kernel_16(const double* A, const double* B, double* C, int i, int j, int M, int K, __mmask8 maskBot, __mmask8 maskTop, int remCols)
{
    __m512d c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, b_pack, a0_pack, a1_pack;

    __m512d zeros = _mm512_setzero_pd();

    if (remCols == 0) return;

    // Top 8 columns
    c0 = _mm512_mask_load_pd(zeros, maskTop, &C[j * M + i]);
    c1 = _mm512_mask_load_pd(zeros, maskTop, &C[(j + 1) * M + i]);
    c2 = _mm512_mask_load_pd(zeros, maskTop, &C[(j + 2) * M + i]);
    c3 = _mm512_mask_load_pd(zeros, maskTop, &C[(j + 3) * M + i]);
    c4 = _mm512_mask_load_pd(zeros, maskTop, &C[(j + 4) * M + i]);
    c5 = _mm512_mask_load_pd(zeros, maskTop, &C[(j + 5) * M + i]);

    // Bottom 8 columns
    c6 = _mm512_mask_load_pd(zeros, maskBot, &C[j * M + (i + 8)]);
    c7 = _mm512_mask_load_pd(zeros, maskBot, &C[(j + 1) * M + (i + 8)]);
    c8 = _mm512_mask_load_pd(zeros, maskBot, &C[(j + 2) * M + (i + 8)]);
    c9 = _mm512_mask_load_pd(zeros, maskBot, &C[(j + 3) * M + (i + 8)]);
    c10 = _mm512_mask_load_pd(zeros, maskBot, &C[(j + 4) * M + (i + 8)]);   
    c11 = _mm512_mask_load_pd(zeros, maskBot, &C[(j + 5) * M + (i + 8)]);

    for (int k = 0; k < K; ++k)
    {
        a0_pack = _mm512_mask_load_pd(zeros, maskTop, &A[k * M + i]);
        a1_pack = _mm512_mask_load_pd(zeros, maskBot, &A[k * M + (i + 8)]);

        b_pack = _mm512_set1_pd(B[j * M + k]);
        c0 = _mm512_fmadd_pd(a0_pack, b_pack, c0);
        c6 = _mm512_fmadd_pd(a1_pack, b_pack, c6);

        if (remCols == 1) continue;
        b_pack = _mm512_set1_pd(B[(j + 1) * M + k]);
        c1 = _mm512_fmadd_pd(a0_pack, b_pack, c1);
        c7 = _mm512_fmadd_pd(a1_pack, b_pack, c7);

        if (remCols == 2) continue;
        b_pack = _mm512_set1_pd(B[(j + 2) * M + k]);
        c2 = _mm512_fmadd_pd(a0_pack, b_pack, c2);
        c8 = _mm512_fmadd_pd(a1_pack, b_pack, c8);

        if (remCols == 3) continue;
        b_pack = _mm512_set1_pd(B[(j + 3) * M + k]);
        c3 = _mm512_fmadd_pd(a0_pack, b_pack, c3);
        c9 = _mm512_fmadd_pd(a1_pack, b_pack, c9);

        if (remCols == 4) continue;
        b_pack = _mm512_set1_pd(B[(j + 4) * M + k]);
        c4 = _mm512_fmadd_pd(a0_pack, b_pack, c4);
        c10 = _mm512_fmadd_pd(a1_pack, b_pack, c10);

        if (remCols == 5) continue;
        b_pack = _mm512_set1_pd(B[(j + 5) * M + k]);
        c5 = _mm512_fmadd_pd(a0_pack, b_pack, c5);
        c11 = _mm512_fmadd_pd(a1_pack, b_pack, c11);
    }

    // Save the C_block back

    _mm512_mask_store_pd(&C[j * M + i], maskTop, c0);
    _mm512_mask_store_pd(&C[(j + 1) * M + i], maskTop, c1);
    _mm512_mask_store_pd(&C[(j + 2) * M + i], maskTop, c2);
    _mm512_mask_store_pd(&C[(j + 3) * M + i], maskTop, c3);
    _mm512_mask_store_pd(&C[(j + 4) * M + i], maskTop, c4);
    _mm512_mask_store_pd(&C[(j + 5) * M + i], maskTop, c5);

    _mm512_mask_store_pd(&C[j * M + (i + 8)], maskBot, c6);
    _mm512_mask_store_pd(&C[(j + 1) * M + (i + 8)], maskBot, c7);
    _mm512_mask_store_pd(&C[(j + 2) * M + (i + 8)], maskBot, c8);
    _mm512_mask_store_pd(&C[(j + 3) * M + (i + 8)], maskBot, c9);
    _mm512_mask_store_pd(&C[(j + 4) * M + (i + 8)], maskBot, c10);
    _mm512_mask_store_pd(&C[(j + 5) * M + (i + 8)], maskBot, c11);
}

void avx512mult(const double *A_block, const double *B_block, double *C_block, int M, int i_end, int j_end, int k_end, int jj, int ii) 
{
    int i, j;

    for (i = 0; i < i_end; i += 16)
    {
        for (j = 0; j < j_end; j += 6)
        {

            const int remCols = M - (j + jj);
            const int remRows = M - (i + ii);
            
            // Create masks
            // Define mask variables for AVX-512
            __mmask8 mask0, mask1; // Another 8-element mask

            // If `m` is not 16, create a mask that only allows the first `m` elements to be loaded
            if (remRows >= 16)
            {
                mask0 = 0xFF;
                mask1 = 0xFF;
            }
            else if (remRows > 8) {
                // Calculate the masks
                mask0 = 0xFF;
                mask1 = (1U << (remRows - 8)) - 1;  // Second mask for the lower half, if `m < 16`
            }
            else
            {
                mask0 = (1U << remRows) - 1;
                mask1 = 0x00;
            }

            micro_kernel_16(A_block, B_block, C_block, i, j, M, k_end, mask1, mask0, remCols);
        }
    }
}


void square_dgemm(const int M, const double *A, const double *B, double *C)
{

    const int block_size1 = 96;
    const int block_size2 = 192;
    // size_t alignment = 64;
    // size_t block_bytes = block_size1 * block_size1 * sizeof(double);

    int ii, jj, kk;

    // // Allocate memory for blocks (block_size x block_size), zero-initialized
    // double* A_block = aligned_alloc(alignment, block_bytes);
    // double* B_block = aligned_alloc(alignment, block_bytes);
    // double* C_block = aligned_alloc(alignment, block_bytes);


    for (jj = 0; jj < M; jj += block_size1)
    {
        for (ii = 0; ii < M; ii += block_size1)
        {
            for (kk = 0; kk < M; kk += block_size2)
            {
                
                int j_end = min(block_size1, M - jj);
                int i_end = min(block_size1, M - ii);
                int k_end = min(block_size2, M - kk);

                // // Copy A_block
                // memset(A_block, 0, block_bytes);
                // for (k = 0; k < k_end; ++k) {
                //     memcpy(&A_block[k * block_size], &A[(kk + k) * M + ii], i_end * sizeof(double));
                // }

                // // Copy B_block
                // memset(B_block, 0, block_bytes);
                // for (j = 0; j < j_end; ++j) {
                //     memcpy(&B_block[j * block_size], &B[(jj + j) * M + kk], k_end * sizeof(double));
                // }

                // // Copy C_block
                // memset(C_block, 0, block_bytes);
                // for (j = 0; j < j_end; ++j) {
                //     memcpy(&C_block[j * block_size], &C[(jj + j) * M + ii], i_end * sizeof(double));
                // }
            
                avx512mult(&A[kk * M + ii], &B[jj * M + kk], &C[jj * M + ii], M, i_end, j_end, k_end, jj, ii);

                // Copy result to C
                // for (j = 0; j < j_end; ++j) {
                //     memcpy(&C[(jj + j) * M + ii], &C_block[j * block_size], i_end * sizeof(double));
                // }
            }
        }
    }
}
