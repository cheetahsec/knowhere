#pragma once

#include "hnswlib.h"
#include "tlsh_utils.h"

namespace hnswlib {

static float
TLSHDistance(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return tlsh::diff((const tlsh::lsh_bin_struct*)pVect1v, (const tlsh::lsh_bin_struct*)pVect2v);
}

class TLSHSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    TLSHSpace(size_t dim) {
        dim_ = dim;
        fstdistfunc_ = TLSHDistance;
        data_size_ = dim / 8;
    }

    size_t 
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> 
    get_dist_func() {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~TLSHSpace() {
    }
};
}  // namespace hnswlib
