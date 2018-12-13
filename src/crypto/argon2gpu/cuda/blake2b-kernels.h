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

#ifndef BLAKE2B_KERNELS_H
#define BLAKE2B_KERNELS_H


enum algo_constants {
    ARGON2_BLOCK_SIZE = 1024,
    ARGON2_QWORDS_IN_BLOCK = ARGON2_BLOCK_SIZE / 8,
    ARGON2_PREHASH_DIGEST_LENGTH = 64,
    ARGON2_PREHASH_SEED_LENGTH = 72,
    BLAKE_BLOCKBYTES = 128
};

enum algo_params {
    ALGO_LANES = 8,
    ALGO_MCOST = 500,
    ALGO_PASSES = 2,
    ALGO_OUTLEN = 32,
    ALGO_VERSION = 0x10,
    ALGO_TOTAL_BLOCKS = (ALGO_MCOST / (4 * ALGO_LANES)) * 4 * ALGO_LANES,
    ALGO_LANE_LENGHT = ALGO_TOTAL_BLOCKS / ALGO_LANES,
    ALGO_SEGMENT_BLOCKS = ALGO_LANE_LENGHT / 4
};

struct partialState {
    uint64_t a, b;
};

struct block {
    uint64_t data[ARGON2_QWORDS_IN_BLOCK];
};

struct block_th {
    uint64_t a, b, c, d;
};

struct uint64x8 {
    uint64_t s0, s1, s2, s3, s4, s5, s6, s7;
};

__device__ __forceinline__
void zero_buffer(uint32_t* buffer, const uint32_t idx) {
    buffer[idx] = 0;
    buffer[idx + 4] = 0;
    buffer[idx + 8] = 0;
    buffer[idx + 12] = 0;
    buffer[idx + 16] = 0;
    buffer[idx + 20] = 0;
    buffer[idx + 24] = 0;
    buffer[idx + 28] = 0;
}

static __constant__   const uint64_t sigma[12][2] = {

    { 506097522914230528,1084818905618843912 },
    { 436021270388410894, 217587900856929281 },
    { 940973067642603531, 290764780619369994 },
    { 1011915791265892615, 580682894302053890 },
    { 1083683067090239497, 937601969488068878 },
    { 218436676723543042, 648815278989708548 },
    { 721716194318550284, 794887571959580416 },
    { 649363922558061325, 721145521830297605 },
    { 576464098234863366, 363107122416517644 },
    { 360576072368521738, 3672381957147407 },
    { 506097522914230528, 1084818905618843912 },
    { 436021270388410894, 217587900856929281 },
};

static __constant__   const uint64_t blake2b_Init[8] = {
    0x6A09E667F2BDC948,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179
};

static __constant__   const uint64_t blake2b_Init_928[8] = {
    0x6A09E667F2BDC928,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179
};

static __constant__   const uint64_t blake2b_IV[8] = {
    7640891576956012808UL,
    13503953896175478587UL,
    4354685564936845355UL,
    11912009170470909681UL,
    5840696475078001361UL,
    11170449401992604703UL,
    2270897969802886507UL,
    6620516959819538809UL
};

__device__  __forceinline__ uint64_t rotate64(const uint64_t x, const uint32_t n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__
void g_shuffle(
    uint64_t* a, uint64_t* b,
    uint64_t* c, uint64_t* d,
    const uint64_t* m1, const uint64_t* m2)
{

    *a = *a + *b + *m1;
    *d = rotate64(*d ^ *a, 32);
    *c = *c + *d;
    *b = rotate64(*b ^ *c, 24);
    *a = *a + *b + *m2;
    *d = rotate64(*d ^ *a, 16);
    *c = *c + *d;
    *b = rotate64(*b ^ *c, 63);

}


__global__ void argon2_initialize_kernel(struct block* memory, uint32_t startNonce);

__global__ void argon2_finalize_kernel(
    block* memory, uint32_t startNonce,
    uint64_t target, uint32_t* resNonces);

__host__ void set_data(const void* data);

#endif
