/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>

namespace faiss {
namespace gpu {

// defines to simplify the SASS assembly structure file/line in the profiler
#define GET_BITFIELD_U32(OUT, VAL, POS, LEN) \
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(OUT) : "r"(VAL), "r"(POS), "r"(LEN));

#define GET_BITFIELD_U64(OUT, VAL, POS, LEN) \
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(OUT) : "l"(VAL), "r"(POS), "r"(LEN));

__device__ __forceinline__ unsigned int getBitfield(
        unsigned int val,
        int pos,
        int len) {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__ uint64_t
getBitfield(uint64_t val, int pos, int len) {
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__ unsigned int setBitfield(
        unsigned int val,
        unsigned int toInsert,
        int pos,
        int len) {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;"
        : "=r"(ret)
        : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

__device__ __forceinline__ unsigned getLaneMaskLt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGt() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ unsigned getLaneMaskGe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ void namedBarrierWait(int name, int numThreads) {
    asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

__device__ __forceinline__ void namedBarrierArrived(int name, int numThreads) {
    asm volatile("bar.arrive %0, %1;"
                 :
                 : "r"(name), "r"(numThreads)
                 : "memory");
}

} // namespace gpu
} // namespace faiss
