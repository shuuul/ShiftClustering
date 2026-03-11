#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "gridshift.h"
#include "utils.h"

namespace nb = nanobind;

static_assert(sizeof(int32_t) == sizeof(int));

NB_MODULE(_gridshift_nb, m) {
    m.def("_grid_cluster", [](int n, int d, int base, int iterations, float bandwidth,
                               nb::ndarray<int32_t, nb::numpy, nb::c_contig, nb::ndim<2>> offsets,
                               nb::ndarray<float, nb::numpy, nb::c_contig, nb::ndim<2>> X_shifted,
                               nb::ndarray<int32_t, nb::numpy, nb::c_contig, nb::ndim<1>> membership,
                               nb::ndarray<int32_t, nb::numpy, nb::c_contig, nb::ndim<1>> k_num) {
        auto *off_ptr = reinterpret_cast<int *>(offsets.data());
        auto *xs_ptr  = X_shifted.data();
        auto *mem_ptr = reinterpret_cast<int *>(membership.data());
        auto *k_ptr   = reinterpret_cast<int *>(k_num.data());
        nb::gil_scoped_release release;
        grid_cluster(n, d, base, iterations, bandwidth, off_ptr, xs_ptr, mem_ptr, k_ptr);
    });

    m.def("_generate_offsets", [](int d, int base,
                                   nb::ndarray<int32_t, nb::numpy, nb::c_contig, nb::ndim<2>> offsets) {
        auto *off_ptr = reinterpret_cast<int *>(offsets.data());
        nb::gil_scoped_release release;
        generate_offsets_cy(d, base, off_ptr);
    });
}
