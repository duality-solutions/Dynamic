/*
 * Copyright (C) 2018-2019 Ehsan Dalvand <dalvand.ehsan@gmail.com>, Alireza Jahandideh <ar.jahandideh@gmail.com>
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation: either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* For IDE: */
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include "crypto/argon2gpu/cuda/kernels.h"
#include "crypto/argon2gpu/cuda/blake2b-kernels.h"

#define INPUT_LEN 80
__constant__ uint32_t d_data[20];

#define CAT(x, y) CAT_(x, y)
#define CAT_(x, y) x ## y

#define G(a,b,c,d,x,col) { \
    ref1=sigma[r][col]>>16*x;\
    ref2=sigma[r][col]>>(16*x+8);\
    CAT(v,a) += CAT(v,b)+m[ref1]; \
    CAT(v,d) = rotate64(CAT(v,d) ^ CAT(v,a),32); \
    CAT(v,c) += CAT(v,d); \
    CAT(v,b) = rotate64(CAT(v,b) ^ CAT(v,c), 24); \
    CAT(v,a) +=CAT(v,b)+m[ref2]; \
    CAT(v,d) = rotate64( CAT(v,d) ^ CAT(v,a), 16); \
    CAT(v,c) += CAT(v,d); \
    CAT(v,b) = rotate64( CAT(v,b) ^ CAT(v,c), 63); \
}

__device__ __forceinline__
void enc32(void *pp, const uint32_t x) {
    uint8_t *p = (uint8_t *) pp;

    p[3] = x & 0xff;
    p[2] = (x >> 8) & 0xff;
    p[1] = (x >> 16) & 0xff;
    p[0] = (x >> 24) & 0xff;
}


__device__ void load_block(uint32_t* dest, uint32_t* src, uint32_t idx) {

    uint32_t i, j;

    for (i = 0; i < 64; i++) {
        j = idx + i * 4;
        dest[j] = src[j];
    }

}

__device__
void blake2b_compress_1w(
    uint64x8* state, const uint64_t* m,
    const uint32_t step, const bool lastChunk = false,
    const size_t lastChunkSize = 0)
{

    uint64_t v0, v1, v2, v3, v4, v5, v6,
             v7, v8, v9, v10, v11, v12,
             v13, v14, v15;

    v0 = state->s0;
    v1 = state->s1;
    v2 = state->s2;
    v3 = state->s3;
    v4 = state->s4;
    v5 = state->s5;
    v6 = state->s6;
    v7 = state->s7;
    v8 = blake2b_IV[0];
    v9 = blake2b_IV[1];
    v10 = blake2b_IV[2];
    v11 = blake2b_IV[3];

    if (lastChunk) {
        v12 = blake2b_IV[4] ^ (step - 1) * BLAKE_BLOCKBYTES + lastChunkSize;
        v14 = blake2b_IV[6] ^ (uint64_t) -1;

    } else {
        v12 = blake2b_IV[4] ^ step * BLAKE_BLOCKBYTES;
        v14 = blake2b_IV[6];
    }

    v13 = blake2b_IV[5];
    v15 = blake2b_IV[7];

#pragma unroll 12
    for (int r = 0; r < 12; r++) {
        uint8_t ref1, ref2;

        /* column step */
        G(0, 4, 8, 12, 0, 0);
        G(1, 5, 9, 13, 1, 0);
        G(2, 6, 10, 14, 2, 0);
        G(3, 7, 11, 15, 3, 0);

        /* diagonal step */
        G(0, 5, 10, 15, 0, 1);
        G(1, 6, 11, 12, 1, 1);
        G(2, 7, 8, 13, 2, 1);
        G(3, 4, 9, 14, 3, 1);
    }

    state->s0 ^= v0 ^ v8;
    state->s1 ^= v1 ^ v9;
    state->s2 ^= v2 ^ v10;
    state->s3 ^= v3 ^ v11;
    state->s4 ^= v4 ^ v12;
    state->s5 ^= v5 ^ v13;
    state->s6 ^= v6 ^ v14;
    state->s7 ^= v7 ^ v15;

}


__device__ void blake2b_compress_4w(
    struct partialState* state, uint64_t* m,
    uint32_t step, uint32_t idx,
    bool lastChunk = false, size_t lastChunkSize = 0)
{

    uint64_t a, b, c, d;

    uint64_t counter = (idx == 0 ? step : 0);

    a = state->a;
    b = state->b;
    c = blake2b_IV[idx];

    if (lastChunk) {
        if (idx == 0)
            d = blake2b_IV[4] ^ (step - 1) * BLAKE_BLOCKBYTES + lastChunkSize;
        else if (idx == 2)
            d = blake2b_IV[6] ^ (uint64_t) -1;
        else
            d = blake2b_IV[idx + 4];
    } else {
        d = blake2b_IV[idx + 4] ^ counter * BLAKE_BLOCKBYTES;
    }

    __syncthreads();

    for (uint32_t r = 0; r < 12; ++r) {

        uint8_t ref1, ref2;

        ref1 = sigma[r][0] >> 8 * 2 * idx;
        ref2 = sigma[r][0] >> 8 * (2 * idx + 1);

        g_shuffle(&a, &b, &c, &d, &m[ref1], &m[ref2]);

        b = __shfl_sync(0xffffffff, b, idx + 1, 4);
        c = __shfl_sync(0xffffffff, c, idx + 2, 4);
        d = __shfl_sync(0xffffffff, d, idx + 3, 4);

        ref1 = sigma[r][1] >> 8 * 2 * idx;
        ref2 = sigma[r][1] >> 8 * (2 * idx + 1);

        g_shuffle(&a, &b, &c, &d, &m[ref1], &m[ref2]);

        b = __shfl_sync(0xffffffff, b, idx - 1, 4);
        c = __shfl_sync(0xffffffff, c, idx - 2, 4);
        d = __shfl_sync(0xffffffff, d, idx - 3, 4);

    }

    state->a = state->a ^ a ^ c;
    state->b = state->b ^ b ^ d;

}


__device__ void computeInitialHash(
    const uint32_t* input, uint32_t* buffer,
    uint32_t nonce)
{

    uint64x8 state;

#pragma unroll
    for (int i = 0; i < 32; i++)
        buffer[i] = 0;

    state.s0 = blake2b_Init[0];
    state.s1 = blake2b_Init[1];
    state.s2 = blake2b_Init[2];
    state.s3 = blake2b_Init[3];
    state.s4 = blake2b_Init[4];
    state.s5 = blake2b_Init[5];
    state.s6 = blake2b_Init[6];
    state.s7 = blake2b_Init[7];

    buffer[0] = ALGO_LANES;
    buffer[1] = ALGO_OUTLEN;
    buffer[2] = ALGO_MCOST;
    buffer[3] = ALGO_PASSES;
    buffer[4] = ALGO_VERSION;
    buffer[6] = 80;

#pragma unroll
    for (int i = 0; i < 19; i++)
        buffer[7 + i] = input[i];

    buffer[26] = nonce;
    buffer[27] = 80;

#pragma unroll
    for (int i = 0; i < 4; i++)
        buffer[28 + i] = input[i];

    blake2b_compress_1w(&state, (uint64_t*) buffer, 1);


#pragma unroll
    for (int i = 0; i < 15; i++)
        buffer[i] = input[i + 4];

    buffer[15] = nonce;

#pragma unroll
    for (int i = 16; i < 32; i++)
        buffer[i] = 0;

    blake2b_compress_1w(&state, (uint64_t*) buffer, 2, true, 72);

#pragma unroll
    for (int i = 0; i < 32; i++)
        buffer[i] = 0;


    memcpy(&buffer[1], &state, 64);

}

__device__ void fillFirstBlock(struct block* memory, uint32_t* buffer) {

    uint32_t row = threadIdx.x / ALGO_LANES;
    uint32_t column = threadIdx.x % ALGO_LANES;

    struct block* memCell = (memory + (blockIdx.x * blockDim.y + threadIdx.y) * ALGO_TOTAL_BLOCKS)
                            + row * ALGO_LANES + column;

    uint64_t* buffer_64 = (uint64_t*) buffer;
    uint64x8 state;

    state.s0 = blake2b_Init[0];
    state.s1 = blake2b_Init[1];
    state.s2 = blake2b_Init[2];
    state.s3 = blake2b_Init[3];
    state.s4 = blake2b_Init[4];
    state.s5 = blake2b_Init[5];
    state.s6 = blake2b_Init[6];
    state.s7 = blake2b_Init[7];

    buffer[0] = 1024;
    buffer[17] = row;
    buffer[18] = column;

    blake2b_compress_1w(&state, buffer_64, 1, true, 76);

    memCell->data[0] = state.s0;
    memCell->data[1] = state.s1;
    memCell->data[2] = state.s2;
    memCell->data[3] = state.s3;

    for (int i = 0; i < 8; i++) {
        buffer_64[i + 8] = 0;
    }

    buffer_64[0] = state.s0;
    buffer_64[1] = state.s1;
    buffer_64[2] = state.s2;
    buffer_64[3] = state.s3;
    buffer_64[4] = state.s4;
    buffer_64[5] = state.s5;
    buffer_64[6] = state.s6;
    buffer_64[7] = state.s7;

    for (uint8_t i = 1; i < 31; i++) {

        state.s0 = blake2b_Init[0];
        state.s1 = blake2b_Init[1];
        state.s2 = blake2b_Init[2];
        state.s3 = blake2b_Init[3];
        state.s4 = blake2b_Init[4];
        state.s5 = blake2b_Init[5];
        state.s6 = blake2b_Init[6];
        state.s7 = blake2b_Init[7];

        blake2b_compress_1w(&state, buffer_64, 1, true, 64);

        buffer_64[0] = state.s0;
        buffer_64[1] = state.s1;
        buffer_64[2] = state.s2;
        buffer_64[3] = state.s3;
        buffer_64[4] = state.s4;
        buffer_64[5] = state.s5;
        buffer_64[6] = state.s6;
        buffer_64[7] = state.s7;

        memCell->data[(i << 2) + 0] = state.s0;
        memCell->data[(i << 2) + 1] = state.s1;
        memCell->data[(i << 2) + 2] = state.s2;
        memCell->data[(i << 2) + 3] = state.s3;

    }

    memCell->data[124] = state.s4;
    memCell->data[125] = state.s5;
    memCell->data[126] = state.s6;
    memCell->data[127] = state.s7;

}

__global__ void argon2_initialize_kernel(struct block* memory, uint32_t startNonce)
{

    uint32_t buffer[32];
    const uint32_t nonce = (blockIdx.x*blockDim.y+threadIdx.y) + startNonce;

    computeInitialHash(d_data, buffer, nonce);
    fillFirstBlock(memory, buffer);

}

__global__ void argon2_finalize_kernel(
    block* memory, uint32_t startNonce,
    uint64_t target, uint32_t* resNonces)
{

    extern __shared__ uint32_t input_t[];
    uint32_t* input = &(input_t[threadIdx.y*258]);
    uint64_t* input_64=(uint64_t*)input;

    uint32_t idx = threadIdx.x;
    uint32_t jobId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t nonce = jobId + startNonce;

    uint32_t* memLane = (uint32_t*) ((memory + jobId * ALGO_TOTAL_BLOCKS));
    partialState state;

    load_block(&input[1], memLane, idx);

    input[0] = 32;

    state.a = blake2b_Init_928[idx];
    state.b = blake2b_Init_928[idx + 4];

    blake2b_compress_4w(&state, &input_64[0], 1, idx);
    blake2b_compress_4w(&state, &input_64[16], 2, idx);
    blake2b_compress_4w(&state, &input_64[32], 3, idx);
    blake2b_compress_4w(&state, &input_64[48], 4, idx);
    blake2b_compress_4w(&state, &input_64[64], 5, idx);
    blake2b_compress_4w(&state, &input_64[80], 6, idx);
    blake2b_compress_4w(&state, &input_64[96], 7, idx);
    blake2b_compress_4w(&state, &input_64[112], 8, idx);

    zero_buffer(&input[0], idx);
    input[0]=input[256];

    blake2b_compress_4w(&state, &input_64[0], 9, idx, true, 4);

    input_64[idx] = state.a;

    __syncthreads();


    if (idx == 0 && input_64[3] <= target) {
        resNonces[0] = nonce;
    }

}

__host__ void set_data(const void* data) {

    cudaMemcpyToSymbol(d_data, data, INPUT_LEN);

}
