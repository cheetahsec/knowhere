/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

//
// This kernel is split into a separate compilation unit to cut down
// on compile time
//

namespace faiss {
namespace gpu {

// This is warp divergence central, but this is really a final step
// and happening a small number of times
//inline __device__ int binarySearchForBucket(
//        int* prefixSumOffsets,
//        int size,
//        int val) {
//    int start = 0;
//    int end = size;
//
//    while (end - start > 0) {
//        int mid = start + (end - start) / 2;
//
//        int midVal = prefixSumOffsets[mid];
//
//        // Find the first bucket that we are <=
//        if (midVal <= val) {
//            start = mid + 1;
//        } else {
//            end = mid;
//        }
//    }
//
//    // We must find the bucket that it is in
//    assert(start != size);
//
//    return start;
//}

template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ, bool Dir>
__global__ void pass2SelectLists(
        Tensor<float, 2, true> heapDistances,
        Tensor<int, 2, true> heapIndices,
        void** listIndices,
        Tensor<int, 2, true> prefixSumOffsets,
        Tensor<int, 2, true> topQueryToCentroid,
        int k,
        IndicesOptions opt,
        Tensor<float, 2, true> outDistances,
        Tensor<Index::idx_t, 2, true> outIndices) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ float smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];

    constexpr auto kInit = Dir ? kFloatMin : kFloatMax;
    BlockSelect<
            float,
            int,
            Dir,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kInit, -1, smemK, smemV, k);

    auto queryId = blockIdx.x;
    int num = heapDistances.getSize(1);
    int limit = utils::roundDown(num, kWarpSize);

    int i = threadIdx.x;
    auto heapDistanceStart = heapDistances[queryId];

    // BlockSelect add cannot be used in a warp divergent circumstance; we
    // handle the remainder warp below
    for (; i < limit; i += blockDim.x) {
        heap.add(heapDistanceStart[i], i);
    }

    // Handle warp divergence separately
    if (i < num) {
        heap.addThreadQ(heapDistanceStart[i], i);
    }

    // Merge all final results
    heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        outDistances[queryId][i] = smemK[i];

        // `v` is the index in `heapIndices`
        // We need to translate this into an original user index. The
        // reason why we don't maintain intermediate results in terms of
        // user indices is to substantially reduce temporary memory
        // requirements and global memory write traffic for the list
        // scanning.
        // This code is highly divergent, but it's probably ok, since this
        // is the very last step and it is happening a small number of
        // times (#queries x k).
        int v = smemV[i];
        Index::idx_t index = -1;

        if (v != -1) {
            // `offset` is the offset of the intermediate result, as
            // calculated by the original scan.
            int offset = heapIndices[queryId][v];

            index = getListIndex(queryId,
                                 offset,
                                 listIndices,
                                 prefixSumOffsets,
                                 topQueryToCentroid,
                                 opt);
        }

        outIndices[queryId][i] = index;
    }
}

void runPass2SelectLists(
        Tensor<float, 2, true>& heapDistances,
        Tensor<int, 2, true>& heapIndices,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<int, 2, true>& topQueryToCentroid,
        int k,
        bool chooseLargest,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices,
        cudaStream_t stream) {
    auto grid = dim3(topQueryToCentroid.getSize(0));

#define RUN_PASS(BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR)         \
    do {                                                       \
        pass2SelectLists<BLOCK, NUM_WARP_Q, NUM_THREAD_Q, DIR> \
                <<<grid, BLOCK, 0, stream>>>(                  \
                        heapDistances,                         \
                        heapIndices,                           \
                        listIndices.data().get(),              \
                        prefixSumOffsets,                      \
                        topQueryToCentroid,                    \
                        k,                                     \
                        indicesOptions,                        \
                        outDistances,                          \
                        outIndices);                           \
        CUDA_TEST_ERROR();                                     \
        return; /* success */                                  \
    } while (0)

#if GPU_MAX_SELECTION_K >= 2048

    // block size 128 for k <= 1024, 64 for k = 2048
#define RUN_PASS_DIR(DIR)                \
    do {                                 \
        if (k == 1) {                    \
            RUN_PASS(128, 1, 1, DIR);    \
        } else if (k <= 32) {            \
            RUN_PASS(128, 32, 2, DIR);   \
        } else if (k <= 64) {            \
            RUN_PASS(128, 64, 3, DIR);   \
        } else if (k <= 128) {           \
            RUN_PASS(128, 128, 3, DIR);  \
        } else if (k <= 256) {           \
            RUN_PASS(128, 256, 4, DIR);  \
        } else if (k <= 512) {           \
            RUN_PASS(128, 512, 8, DIR);  \
        } else if (k <= 1024) {          \
            RUN_PASS(128, 1024, 8, DIR); \
        } else if (k <= 2048) {          \
            RUN_PASS(64, 2048, 8, DIR);  \
        }                                \
    } while (0)

#else

#define RUN_PASS_DIR(DIR)                \
    do {                                 \
        if (k == 1) {                    \
            RUN_PASS(128, 1, 1, DIR);    \
        } else if (k <= 32) {            \
            RUN_PASS(128, 32, 2, DIR);   \
        } else if (k <= 64) {            \
            RUN_PASS(128, 64, 3, DIR);   \
        } else if (k <= 128) {           \
            RUN_PASS(128, 128, 3, DIR);  \
        } else if (k <= 256) {           \
            RUN_PASS(128, 256, 4, DIR);  \
        } else if (k <= 512) {           \
            RUN_PASS(128, 512, 8, DIR);  \
        } else if (k <= 1024) {          \
            RUN_PASS(128, 1024, 8, DIR); \
        }                                \
    } while (0)

#endif // GPU_MAX_SELECTION_K

    if (chooseLargest) {
        RUN_PASS_DIR(true);
    } else {
        RUN_PASS_DIR(false);
    }

    // unimplemented / too many resources
    FAISS_ASSERT_FMT(false, "unimplemented k value (%d)", k);

#undef RUN_PASS_DIR
#undef RUN_PASS
}

} // namespace gpu
} // namespace faiss
