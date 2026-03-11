#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "localshift.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

static void localshift_parallel(float *points, int cnt,
                                int n_steps, float fmaxd, float fsiv, float tol,
                                int ref_shape_0, int ref_shape_1, int ref_shape_2,
                                float *reference, int num_threads) {
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < cnt; i++) {
        localshift_single_cy(&points[i * 3], n_steps, fmaxd, fsiv, tol,
                              ref_shape_0, ref_shape_1, ref_shape_2, reference);
    }
}

NB_MODULE(_localshift_nb, m) {
    m.def("_localshift_parallel",
          [](nb::ndarray<float, nb::numpy, nb::c_contig, nb::ndim<2>> points,
             nb::ndarray<float, nb::numpy, nb::c_contig, nb::ndim<3>> reference,
             float fmaxd, float fsiv, int n_steps, float tol, int num_threads) {
              int cnt = static_cast<int>(points.shape(0));
              int ref0 = static_cast<int>(reference.shape(0));
              int ref1 = static_cast<int>(reference.shape(1));
              int ref2 = static_cast<int>(reference.shape(2));
              float *pts_ptr = points.data();
              float *ref_ptr = reference.data();
              nb::gil_scoped_release release;
              localshift_parallel(pts_ptr, cnt, n_steps, fmaxd, fsiv, tol,
                                  ref0, ref1, ref2, ref_ptr, num_threads);
          });
}
