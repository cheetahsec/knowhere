# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_definitions(-DKNOWHERE_WITH_RAFT)
include(cmake/utils/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

set(RAPIDS_VERSION 23.04)
set(RAFT_VERSION "${RAPIDS_VERSION}")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAPIDS_VERSION}")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  # -----------------------------------------------------
  # Invoke CPM find_package()
  # -----------------------------------------------------
  rapids_cpm_find(
    raft
    ${PKG_VERSION}
    GLOBAL_TARGETS
    raft::raft
    COMPONENTS
    ${RAFT_COMPONENTS}
    CPM_ARGS
    GIT_REPOSITORY
    https://github.com/${PKG_FORK}/raft.git
    GIT_TAG
    ${PKG_PINNED_TAG}
    SOURCE_SUBDIR
    cpp
    OPTIONS
    "BUILD_TESTS OFF"
    "BUILD_BENCH OFF"
    "RAFT_USE_FAISS_STATIC OFF") # Turn this on to build FAISS into your binary

    if(raft_ADDED)
        message(VERBOSE "KNOWHERE: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "KNOWHERE: Using RAFT located in ${raft_DIR}")
    endif()
endfunction()

# Change pinned tag here to test a commit in CI To use a different RAFT locally,
# set the CMake variable CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION ${RAFT_VERSION}.00 FORK ${RAFT_FORK} PINNED_TAG
                        ${RAFT_PINNED_TAG} COMPILE_LIBRARY OFF)
