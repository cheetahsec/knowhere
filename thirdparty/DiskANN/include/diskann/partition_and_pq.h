// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include "neighbor.h"
#include "parameters.h"
#include "tsl/robin_set.h"
#include "utils.h"

#include "windows_customizations.h"

template<typename T>
void gen_random_slice(const std::string base_file,
                      const std::string output_prefix, double sampling_rate);

template<typename T>
void gen_random_slice(const std::string data_file, double p_val,
                      float *&sampled_data, size_t &slice_size, size_t &ndims);

template<typename T>
void gen_random_slice(const T *inputdata, size_t npts, size_t ndims,
                      double p_val, float *&sampled_data, size_t &slice_size);

int estimate_cluster_sizes(float *test_data_float, size_t num_test,
                           float *pivots, const size_t num_centers,
                           const size_t dim, const size_t k_base,
                           std::vector<size_t> &cluster_sizes);

template<typename T>
int shard_data_into_clusters(const std::string data_file, float *pivots,
                             const size_t num_centers, const size_t dim,
                             const size_t k_base, std::string prefix_path);

template<typename T>
int shard_data_into_clusters_only_ids(const std::string data_file,
                                      float *pivots, const size_t num_centers,
                                      const size_t dim, const size_t k_base,
                                      std::string prefix_path);

template<typename T>
int retrieve_shard_data_from_ids(const std::string data_file,
                                 std::string       idmap_filename,
                                 std::string       data_filename);

template<typename T>
int partition(const std::string data_file, const float sampling_rate,
              size_t num_centers, size_t max_k_means_reps,
              const std::string prefix_path, size_t k_base);

template<typename T>
int partition_with_ram_budget(const std::string data_file,
                              const double sampling_rate, double ram_budget,
                              size_t            graph_degree,
                              const std::string prefix_path, size_t k_base);

DISKANN_DLLEXPORT int generate_pq_pivots(
    const float *train_data, size_t num_train, unsigned dim,
    unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
    std::string pq_pivots_path, bool make_zero_mean = false);

template<typename T>
int generate_pq_data_from_pivots(const std::string data_file,
                                 unsigned num_centers, unsigned num_pq_chunks,
                                 std::string pq_pivots_path,
                                 std::string pq_compressed_vectors_path);
