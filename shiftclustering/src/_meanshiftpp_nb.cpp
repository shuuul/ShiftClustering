#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "meanshiftpp.h"
#include "utils.h"

namespace nb = nanobind;

static_assert(sizeof(int32_t) == sizeof(int));

NB_MODULE(_meanshiftpp_nb, m) {
    m.def("_shift_cy", [](int n, int d, int base, float bandwidth,
                           nb::ndarray<int32_t, nb::numpy, nb::c_contig, nb::ndim<2>> offsets,
                           nb::ndarray<float, nb::numpy, nb::c_contig, nb::ndim<2>> X_shifted) {
        auto *off_ptr = reinterpret_cast<int *>(offsets.data());
        auto *xs_ptr  = X_shifted.data();
        nb::gil_scoped_release release;
        shift_cy(n, d, base, bandwidth, off_ptr, xs_ptr);
    });

    m.def("_generate_offsets", [](int d, int base,
                                   nb::ndarray<int32_t, nb::numpy, nb::c_contig, nb::ndim<2>> offsets) {
        auto *off_ptr = reinterpret_cast<int *>(offsets.data());
        nb::gil_scoped_release release;
        generate_offsets_cy(d, base, off_ptr);
    });
}
