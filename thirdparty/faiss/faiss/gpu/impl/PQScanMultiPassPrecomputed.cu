/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/gpu/impl/IVFUtils.cuh>
#include <faiss/gpu/impl/PQCodeLoad.cuh>
#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>
#include <faiss/gpu/utils/LoadStoreOperators.cuh>
#include <faiss/gpu/utils/MathOperators.cuh>
#include <faiss/gpu/utils/WarpPackedBits.cuh>

#include <algorithm>
#include <limits>

namespace faiss {
namespace gpu {

// A basic implementation that works for the interleaved by vector layout for
// any number of sub-quantizers
template <typename EncodeT, int EncodeBits, typename CodeDistanceT>
__global__ void pqScanPrecomputedInterleaved(
        Tensor<float, 2, true> queries,
        // (query id)(probe id)
        Tensor<float, 2, true> precompTerm1,
        // (centroid id)(sub q)(code id)
        Tensor<CodeDistanceT, 3, true> precompTerm2,
        // (query id)(sub q)(code id)
        Tensor<CodeDistanceT, 3, true> precompTerm3,
        Tensor<int, 2, true> topQueryToCentroid,
        void** listCodes,
        int* listLengths,
        Tensor<int, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    // Each block handles a single query versus single list
    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;

    auto listId = topQueryToCentroid[queryId][probeId];
    // Safety guard in case NaNs in input cause no list ID to be generated
    if (listId == -1) {
        return;
    }

    int numWarps = blockDim.x / kWarpSize;
    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
    int laneId = threadIdx.x % kWarpSize;
    int warpId = threadIdx.x / kWarpSize;

    auto numSubQuantizers = precompTerm2.getSize(1);
    auto codesPerSubQuantizer = precompTerm2.getSize(2);

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value
    int outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);
    float* distanceOut = distance[outBase].data();

    auto vecsBase = (EncodeT*)listCodes[listId];
    int numVecs = listLengths[listId];

    // How many vector blocks of 32 are in this list?
    int numBlocks = utils::divUp(numVecs, 32);

    // Number of EncodeT words per each dimension of block of 32 vecs
    constexpr int bytesPerVectorBlockDim = EncodeBits * 32 / 8;
    constexpr int wordsPerVectorBlockDim =
            bytesPerVectorBlockDim / sizeof(EncodeT);
    int wordsPerVectorBlock = wordsPerVectorBlockDim * numSubQuantizers;

    // This is constant for the (query, probe)
    float term1 = precompTerm1[queryId][probeId];

    for (int block = warpId; block < numBlocks; block += numWarps) {
        float dist = term1;

        // This is the vector a given lane/thread handles
        int vec = block * kWarpSize + laneId;
        bool valid = vec < numVecs;

        EncodeT* data = vecsBase + block * wordsPerVectorBlock;
        auto term2Base = precompTerm2[listId].data();
        auto term3Base = precompTerm3[queryId].data();

        for (int sq = 0; sq < numSubQuantizers; ++sq) {
            EncodeT enc =
                    WarpPackedBits<EncodeT, EncodeBits>::read(laneId, data);
            EncodeT code =
                    WarpPackedBits<EncodeT, EncodeBits>::postRead(laneId, enc);

            dist += valid ? (ConvertTo<float>::to(term2Base[code]) +
                             ConvertTo<float>::to(term3Base[code]))
                          : 0;

            data += wordsPerVectorBlockDim;
            term2Base += codesPerSubQuantizer;
            term3Base += codesPerSubQuantizer;
        }

        if (valid) {
            distanceOut[vec] = dist;
        }
    }
}

// For precomputed codes, this calculates and loads code distances
// into smem
template <typename LookupT, typename LookupVecT>
inline __device__ void loadPrecomputedTerm(
        LookupT* smem,
        LookupT* term2Start,
        LookupT* term3Start,
        int numCodes) {
    constexpr int kWordSize = sizeof(LookupVecT) / sizeof(LookupT);

    // We can only use vector loads if the data is guaranteed to be
    // aligned. The codes are innermost, so if it is evenly divisible,
    // then any slice will be aligned.
    if (numCodes % kWordSize == 0) {
        constexpr int kUnroll = 2;

        // Load the data by float4 for efficiency, and then handle any remainder
        // limitVec is the number of whole vec words we can load, in terms
        // of whole blocks performing the load
        int limitVec = numCodes / (kUnroll * kWordSize * blockDim.x);
        limitVec *= kUnroll * blockDim.x;

        LookupVecT* smemV = (LookupVecT*)smem;
        LookupVecT* term2StartV = (LookupVecT*)term2Start;
        LookupVecT* term3StartV = (LookupVecT*)term3Start;

        for (int i = threadIdx.x; i < limitVec; i += kUnroll * blockDim.x) {
            LookupVecT vals[kUnroll];

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                vals[j] = LoadStore<LookupVecT>::load(
                        &term2StartV[i + j * blockDim.x]);
            }

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                LookupVecT q = LoadStore<LookupVecT>::load(
                        &term3StartV[i + j * blockDim.x]);

                vals[j] = Math<LookupVecT>::add(vals[j], q);
            }

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                LoadStore<LookupVecT>::store(
                        &smemV[i + j * blockDim.x], vals[j]);
            }
        }

        // This is where we start loading the remainder that does not evenly
        // fit into kUnroll x blockDim.x
        int remainder = limitVec * kWordSize;

        for (int i = remainder + threadIdx.x; i < numCodes; i += blockDim.x) {
            smem[i] = Math<LookupT>::add(term2Start[i], term3Start[i]);
        }
    } else {
        // Potential unaligned load
        constexpr int kUnroll = 4;

        int limit = utils::roundDown(numCodes, kUnroll * blockDim.x);

        int i = threadIdx.x;
        for (; i < limit; i += kUnroll * blockDim.x) {
            LookupT vals[kUnroll];

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                vals[j] = term2Start[i + j * blockDim.x];
            }

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                vals[j] = Math<LookupT>::add(
                        vals[j], term3Start[i + j * blockDim.x]);
            }

#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                smem[i + j * blockDim.x] = vals[j];
            }
        }

        for (; i < numCodes; i += blockDim.x) {
            smem[i] = Math<LookupT>::add(term2Start[i], term3Start[i]);
        }
    }
}

template <int NumSubQuantizers, typename LookupT, typename LookupVecT>
__global__ void pqScanPrecomputedMultiPass(
        Tensor<float, 2, true> queries,
        Tensor<float, 2, true> precompTerm1,
        Tensor<LookupT, 3, true> precompTerm2,
        Tensor<LookupT, 3, true> precompTerm3,
        Tensor<int, 2, true> topQueryToCentroid,
        void** listCodes,
        int* listLengths,
        Tensor<int, 2, true> prefixSumOffsets,
        Tensor<float, 1, true> distance) {
    // precomputed term 2 + 3 storage
    // (sub q)(code id)
    extern __shared__ char smemTerm23[];
    LookupT* term23 = (LookupT*)smemTerm23;

    // Each block handles a single query
    auto queryId = blockIdx.y;
    auto probeId = blockIdx.x;
    auto codesPerSubQuantizer = precompTerm2.getSize(2);
    auto precompTermSize = precompTerm2.getSize(1) * codesPerSubQuantizer;

    // This is where we start writing out data
    // We ensure that before the array (at offset -1), there is a 0 value
    int outBase = *(prefixSumOffsets[queryId][probeId].data() - 1);
    float* distanceOut = distance[outBase].data();

    auto listId = topQueryToCentroid[queryId][probeId];
    // Safety guard in case NaNs in input cause no list ID to be generated
    if (listId == -1) {
        return;
    }

    uint8_t* codeList = (uint8_t*)listCodes[listId];
    int limit = listLengths[listId];

    constexpr int kNumCode32 =
            NumSubQuantizers <= 4 ? 1 : (NumSubQuantizers / 4);
    unsigned int code32[kNumCode32];
    unsigned int nextCode32[kNumCode32];

    // We double-buffer the code loading, which improves memory utilization
    if (threadIdx.x < limit) {
        LoadCode32<NumSubQuantizers>::load(code32, codeList, threadIdx.x);
    }

    // Load precomputed terms 1, 2, 3
    float term1 = precompTerm1[queryId][probeId];
    loadPrecomputedTerm<LookupT, LookupVecT>(
            term23,
            precompTerm2[listId].data(),
            precompTerm3[queryId].data(),
            precompTermSize);

    // Prevent WAR dependencies
    __syncthreads();

    // Each thread handles one code element in the list, with a
    // block-wide stride
    for (int codeIndex = threadIdx.x; codeIndex < limit;
         codeIndex += blockDim.x) {
        // Prefetch next codes
        if (codeIndex + blockDim.x < limit) {
            LoadCode32<NumSubQuantizers>::load(
                    nextCode32, codeList, codeIndex + blockDim.x);
        }

        float dist = term1;

#pragma unroll
        for (int word = 0; word < kNumCode32; ++word) {
            constexpr int kBytesPerCode32 =
                    NumSubQuantizers < 4 ? NumSubQuantizers : 4;

            if (kBytesPerCode32 == 1) {
                auto code = code32[0];
                dist = ConvertTo<float>::to(term23[code]);

            } else {
#pragma unroll
                for (int byte = 0; byte < kBytesPerCode32; ++byte) {
                    auto code = getByte(code32[word], byte * 8, 8);

                    auto offset = codesPerSubQuantizer *
                            (word * kBytesPerCode32 + byte);

                    dist += ConvertTo<float>::to(term23[offset + code]);
                }
            }
        }

        // Write out intermediate distance result
        // We do not maintain indices here, in order to reduce global
        // memory traffic. Those are recovered in the final selection step.
        distanceOut[codeIndex] = dist;

        // Rotate buffers
#pragma unroll
        for (int word = 0; word < kNumCode32; ++word) {
            code32[word] = nextCode32[word];
        }
    }
}

void runMultiPassTile(
        GpuResources* res,
        Tensor<float, 2, true>& queries,
        Tensor<float, 2, true>& precompTerm1,
        NoTypeTensor<3, true>& precompTerm2,
        NoTypeTensor<3, true>& precompTerm3,
        Tensor<int, 2, true>& topQueryToCentroid,
        Tensor<uint8_t, 1, true>& bitset,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        thrust::device_vector<void*>& listCodes,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        thrust::device_vector<int>& listLengths,
        Tensor<char, 1, true>& thrustMem,
        Tensor<int, 2, true>& prefixSumOffsets,
        Tensor<float, 1, true>& allDistances,
        Tensor<float, 3, true>& heapDistances,
        Tensor<int, 3, true>& heapIndices,
        int k,
        Tensor<float, 2, true>& outDistances,
        Tensor<Index::idx_t, 2, true>& outIndices,
        cudaStream_t stream) {
    // Calculate offset lengths, so we know where to write out
    // intermediate results
    runCalcListOffsets(
            res,
            topQueryToCentroid,
            listLengths,
            prefixSumOffsets,
            thrustMem,
            stream);

    // The vector interleaved layout implementation
    if (interleavedCodeLayout) {
        auto kThreadsPerBlock = 256;

        auto grid = dim3(
                topQueryToCentroid.getSize(1), topQueryToCentroid.getSize(0));
        auto block = dim3(kThreadsPerBlock);

#define RUN_INTERLEAVED(BITS_PER_CODE, CODE_DIST_T)                       \
    do {                                                                  \
        pqScanPrecomputedInterleaved<uint8_t, BITS_PER_CODE, CODE_DIST_T> \
                <<<grid, block, 0, stream>>>(                             \
                        queries,                                          \
                        precompTerm1,                                     \
                        precompTerm2T,                                    \
                        precompTerm3T,                                    \
                        topQueryToCentroid,                               \
                        listCodes.data().get(),                           \
                        listLengths.data().get(),                         \
                        prefixSumOffsets,                                 \
                        allDistances);                                    \
    } while (0)

        if (useFloat16Lookup) {
            auto precompTerm2T = precompTerm2.toTensor<half>();
            auto precompTerm3T = precompTerm3.toTensor<half>();

            switch (bitsPerSubQuantizer) {
                case 4: {
                    RUN_INTERLEAVED(4, half);
                } break;
                case 5: {
                    RUN_INTERLEAVED(5, half);
                } break;
                case 6: {
                    RUN_INTERLEAVED(6, half);
                } break;
                case 8: {
                    RUN_INTERLEAVED(8, half);
                } break;
                default:
                    FAISS_ASSERT(false);
                    break;
            }
        } else {
            auto precompTerm2T = precompTerm2.toTensor<float>();
            auto precompTerm3T = precompTerm3.toTensor<float>();

            switch (bitsPerSubQuantizer) {
                case 4: {
                    RUN_INTERLEAVED(4, float);
                } break;
                case 5: {
                    RUN_INTERLEAVED(5, float);
                } break;
                case 6: {
                    RUN_INTERLEAVED(6, float);
                } break;
                case 8: {
                    RUN_INTERLEAVED(8, float);
                } break;
                default:
                    FAISS_ASSERT(false);
                    break;
            }
        }
    } else {
        // Convert all codes to a distance, and write out (distance,
        // index) values for all intermediate results
        auto kThreadsPerBlock = 256;

        auto grid = dim3(
                topQueryToCentroid.getSize(1), topQueryToCentroid.getSize(0));
        auto block = dim3(kThreadsPerBlock);

        // pq precomputed terms (2 + 3)
        auto smem = useFloat16Lookup ? sizeof(half) : sizeof(float);

        smem *= numSubQuantizers * numSubQuantizerCodes;
        FAISS_ASSERT(smem <= getMaxSharedMemPerBlockCurrentDevice());

#define RUN_PQ_OPT(NUM_SUB_Q, LOOKUP_T, LOOKUP_VEC_T)                 \
    do {                                                              \
        auto precompTerm2T = precompTerm2.toTensor<LOOKUP_T>();       \
        auto precompTerm3T = precompTerm3.toTensor<LOOKUP_T>();       \
                                                                      \
        pqScanPrecomputedMultiPass<NUM_SUB_Q, LOOKUP_T, LOOKUP_VEC_T> \
                <<<grid, block, smem, stream>>>(                      \
                        queries,                                      \
                        precompTerm1,                                 \
                        precompTerm2T,                                \
                        precompTerm3T,                                \
                        topQueryToCentroid,                           \
                        listCodes.data().get(),                       \
                        listLengths.data().get(),                     \
                        prefixSumOffsets,                             \
                        allDistances);                                \
    } while (0)

#define RUN_PQ(NUM_SUB_Q)                         \
    do {                                          \
        if (useFloat16Lookup) {                   \
            RUN_PQ_OPT(NUM_SUB_Q, half, Half8);   \
        } else {                                  \
            RUN_PQ_OPT(NUM_SUB_Q, float, float4); \
        }                                         \
    } while (0)

        switch (numSubQuantizers) {
            case 1:
                RUN_PQ(1);
                break;
            case 2:
                RUN_PQ(2);
                break;
            case 3:
                RUN_PQ(3);
                break;
            case 4:
                RUN_PQ(4);
                break;
            case 8:
                RUN_PQ(8);
                break;
            case 12:
                RUN_PQ(12);
                break;
            case 16:
                RUN_PQ(16);
                break;
            case 20:
                RUN_PQ(20);
                break;
            case 24:
                RUN_PQ(24);
                break;
            case 28:
                RUN_PQ(28);
                break;
            case 32:
                RUN_PQ(32);
                break;
            case 40:
                RUN_PQ(40);
                break;
            case 48:
                RUN_PQ(48);
                break;
            case 56:
                RUN_PQ(56);
                break;
            case 64:
                RUN_PQ(64);
                break;
            case 96:
                RUN_PQ(96);
                break;
            default:
                FAISS_ASSERT(false);
                break;
        }

        CUDA_TEST_ERROR();

#undef RUN_PQ
#undef RUN_PQ_OPT
#undef RUN_INTERLEAVED
    }

    // k-select the output in chunks, to increase parallelism
    runPass1SelectLists(
            listIndices,
            indicesOptions,
            prefixSumOffsets,
            topQueryToCentroid,
            bitset,
            allDistances,
            topQueryToCentroid.getSize(1),
            k,
            false, // L2 distance chooses smallest
            heapDistances,
            heapIndices,
            stream);

    // k-select final output
    auto flatHeapDistances = heapDistances.downcastInner<2>();
    auto flatHeapIndices = heapIndices.downcastInner<2>();

    runPass2SelectLists(
            flatHeapDistances,
            flatHeapIndices,
            listIndices,
            indicesOptions,
            prefixSumOffsets,
            topQueryToCentroid,
            k,
            false, // L2 distance chooses smallest
            outDistances,
            outIndices,
            stream);

    CUDA_TEST_ERROR();
}

void runPQScanMultiPassPrecomputed(
        Tensor<float, 2, true>& queries,
        // (query id)(probe id)
        Tensor<float, 2, true>& precompTerm1,
        // (centroid id)(sub q)(code id)
        NoTypeTensor<3, true>& precompTerm2,
        // (query id)(sub q)(code id)
        NoTypeTensor<3, true>& precompTerm3,
        Tensor<int, 2, true>& topQueryToCentroid,
        Tensor<uint8_t, 1, true>& bitset,
        bool useFloat16Lookup,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        thrust::device_vector<void*>& listCodes,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        thrust::device_vector<int>& listLengths,
        int maxListLength,
        int k,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<Index::idx_t, 2, true>& outIndices,
        GpuResources* res) {
    constexpr int kMinQueryTileSize = 8;
    constexpr int kMaxQueryTileSize = 128;
    constexpr int kThrustMemSize = 16384;

    int nprobe = topQueryToCentroid.getSize(1);

    auto stream = res->getDefaultStreamCurrentDevice();

    // Make a reservation for Thrust to do its dirty work (global memory
    // cross-block reduction space); hopefully this is large enough.
    DeviceTensor<char, 1, true> thrustMem1(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true> thrustMem2(
            res, makeTempAlloc(AllocType::Other, stream), {kThrustMemSize});
    DeviceTensor<char, 1, true>* thrustMem[2] = {&thrustMem1, &thrustMem2};

    // How much temporary storage is available?
    // If possible, we'd like to fit within the space available.
    size_t sizeAvailable = res->getTempMemoryAvailableCurrentDevice();

    // We run two passes of heap selection
    // This is the size of the first-level heap passes
    constexpr int kNProbeSplit = 8;
    int pass2Chunks = std::min(nprobe, kNProbeSplit);

    size_t sizeForFirstSelectPass =
            pass2Chunks * k * (sizeof(float) + sizeof(int));

    // How much temporary storage we need per each query
    size_t sizePerQuery = 2 *                         // # streams
            ((nprobe * sizeof(int) + sizeof(int)) +   // prefixSumOffsets
             nprobe * maxListLength * sizeof(float) + // allDistances
             sizeForFirstSelectPass);

    int queryTileSize = (int)(sizeAvailable / sizePerQuery);

    if (queryTileSize < kMinQueryTileSize) {
        queryTileSize = kMinQueryTileSize;
    } else if (queryTileSize > kMaxQueryTileSize) {
        queryTileSize = kMaxQueryTileSize;
    }

    // FIXME: we should adjust queryTileSize to deal with this, since
    // indexing is in int32
    FAISS_ASSERT(
            queryTileSize * nprobe * maxListLength <=
            std::numeric_limits<int>::max());

    // Temporary memory buffers
    // Make sure there is space prior to the start which will be 0, and
    // will handle the boundary condition without branches
    DeviceTensor<int, 1, true> prefixSumOffsetSpace1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});
    DeviceTensor<int, 1, true> prefixSumOffsetSpace2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe + 1});

    DeviceTensor<int, 2, true> prefixSumOffsets1(
            prefixSumOffsetSpace1[1].data(), {queryTileSize, nprobe});
    DeviceTensor<int, 2, true> prefixSumOffsets2(
            prefixSumOffsetSpace2[1].data(), {queryTileSize, nprobe});
    DeviceTensor<int, 2, true>* prefixSumOffsets[2] = {
            &prefixSumOffsets1, &prefixSumOffsets2};

    // Make sure the element before prefixSumOffsets is 0, since we
    // depend upon simple, boundary-less indexing to get proper results
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace1.data(), 0, sizeof(int), stream));
    CUDA_VERIFY(cudaMemsetAsync(
            prefixSumOffsetSpace2.data(), 0, sizeof(int), stream));

    DeviceTensor<float, 1, true> allDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true> allDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize * nprobe * maxListLength});
    DeviceTensor<float, 1, true>* allDistances[2] = {
            &allDistances1, &allDistances2};

    DeviceTensor<float, 3, true> heapDistances1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true> heapDistances2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<float, 3, true>* heapDistances[2] = {
            &heapDistances1, &heapDistances2};

    DeviceTensor<int, 3, true> heapIndices1(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<int, 3, true> heapIndices2(
            res,
            makeTempAlloc(AllocType::Other, stream),
            {queryTileSize, pass2Chunks, k});
    DeviceTensor<int, 3, true>* heapIndices[2] = {&heapIndices1, &heapIndices2};

    auto streams = res->getAlternateStreamsCurrentDevice();
    streamWait(streams, {stream});

    int curStream = 0;

    for (int query = 0; query < queries.getSize(0); query += queryTileSize) {
        int numQueriesInTile =
                std::min(queryTileSize, queries.getSize(0) - query);

        auto prefixSumOffsetsView =
                prefixSumOffsets[curStream]->narrowOutermost(
                        0, numQueriesInTile);

        auto coarseIndicesView =
                topQueryToCentroid.narrowOutermost(query, numQueriesInTile);
        auto queryView = queries.narrowOutermost(query, numQueriesInTile);
        auto term1View = precompTerm1.narrowOutermost(query, numQueriesInTile);
        auto term3View = precompTerm3.narrowOutermost(query, numQueriesInTile);

        auto heapDistancesView =
                heapDistances[curStream]->narrowOutermost(0, numQueriesInTile);
        auto heapIndicesView =
                heapIndices[curStream]->narrowOutermost(0, numQueriesInTile);

        auto outDistanceView =
                outDistances.narrowOutermost(query, numQueriesInTile);
        auto outIndicesView =
                outIndices.narrowOutermost(query, numQueriesInTile);

        runMultiPassTile(
                res,
                queryView,
                term1View,
                precompTerm2,
                term3View,
                coarseIndicesView,
                bitset,
                useFloat16Lookup,
                interleavedCodeLayout,
                bitsPerSubQuantizer,
                numSubQuantizers,
                numSubQuantizerCodes,
                listCodes,
                listIndices,
                indicesOptions,
                listLengths,
                *thrustMem[curStream],
                prefixSumOffsetsView,
                *allDistances[curStream],
                heapDistancesView,
                heapIndicesView,
                k,
                outDistanceView,
                outIndicesView,
                streams[curStream]);

        curStream = (curStream + 1) % 2;
    }

    streamWait({stream}, streams);
}

} // namespace gpu
} // namespace faiss
