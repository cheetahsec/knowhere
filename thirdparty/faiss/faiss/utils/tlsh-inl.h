// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

#ifndef FAISS_TLSH_INL_H
#define FAISS_TLSH_INL_H

#include "hnswlib/tlsh_utils.h"

namespace faiss {

struct TLSHComputerDefault {
    const uint8_t* a;
    int n;

    TLSHComputerDefault() {}

    TLSHComputerDefault(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        a = a8;
        n = code_size;
    }

    float compute(const uint8_t* b8) const {
        return tlsh::diff((const tlsh::lsh_bin_struct*)a, (const tlsh::lsh_bin_struct*)b8);
    }
};

} // namespace faiss

#endif
