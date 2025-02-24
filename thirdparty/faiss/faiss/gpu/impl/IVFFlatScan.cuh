/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/GpuIndicesOptions.h>
#include <thrust/device_vector.h>
#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
#include <faiss/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

void runIVFFlatScan(
        Tensor<float, 2, true>& queries,
        Tensor<int, 2, true>& listIds,
        Tensor<uint8_t, 1, true>& bitset,
        thrust::device_vector<void*>& listData,
        thrust::device_vector<void*>& listIndices,
        IndicesOptions indicesOptions,
        thrust::device_vector<int>& listLengths,
        int maxListLength,
        int k,
        faiss::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<Index::idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss
