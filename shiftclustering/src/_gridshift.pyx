#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3

from collections import Counter

cimport numpy as cnp
import numpy as np

cnp.import_array()

ctypedef cnp.int32_t int32_t
ctypedef cnp.float32_t float32_t


cdef extern from "gridshift.h":
    void grid_cluster(int n,
                      int d,
                      int base,
                      int iterations,
                      float bandwidth,
                      int * offsets,
                      float * X_shifted,
                      int * membership,
                      int * k_num) nogil

cdef extern from "utils.h":
    void generate_offsets_cy(int d,
                             int base,
                             int * offsets) nogil


cdef void generate_offsets_np(int d,
                              int base,
                              int32_t[:, ::1] offsets) nogil:
    generate_offsets_cy(d, base, <int*>&offsets[0, 0])


class GridShift:
    """
    Parameters
    ----------

    bandwidth: Radius for binning points. Points are assigned to the bin
               corresponding to floor division by bandwidth

    threshold: Stop shifting if the L2 norm between max_iters is less than
               threshold

    max_iters: Maximum number of max_iters to run

    """

    def __init__(self, bandwidth, threshold=1.0e-4, max_iters=300):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.max_iters = max_iters

    def fit_predict(self, X, return_centers=False):
        """
        Determines the clusters in either `max_iters` or when the L2
        norm of consecutive max_iters is less than `threshold`, whichever
        comes first.
        Each shift has two steps: First, points are binned based on floor
        division by bandwidth. Second, each bin is shifted to the
        weighted mean of its 3**d neighbors.
        Lastly, points that are in the same bin are clustered together.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint in
           Euclidean space

        Returns
        ----------
        (n, ) cluster labels
        """

        X = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        X_shifted = np.copy(X)
        membership = np.full(n, -1, dtype=np.int32)

        base = 3
        offsets = np.full((base ** d, d), -1, dtype=np.int32)
        k = np.full((1,), -1, dtype=np.int32)
        
        # Create memory views for efficient access
        cdef int32_t[:, ::1] offsets_view = offsets
        cdef float32_t[:, ::1] X_shifted_view = X_shifted
        cdef int32_t[::1] membership_view = membership
        cdef int32_t[::1] k_view = k
        cdef int c_n = n
        cdef int c_d = d
        cdef int c_base = base
        cdef int c_max_iters = self.max_iters
        cdef float c_bandwidth = self.bandwidth
        
        generate_offsets_np(d, base, offsets_view)

        # Use sequential grid clustering
        with nogil:
            grid_cluster(c_n, c_d, c_base, c_max_iters, c_bandwidth,
                        <int*>&offsets_view[0, 0], <float*>&X_shifted_view[0, 0],
                        <int*>&membership_view[0], <int*>&k_view[0])

        cluster_centers = X_shifted[0:np.ndarray.item(k), :]
        if return_centers:
            return cluster_centers, membership
        else:
            return membership
