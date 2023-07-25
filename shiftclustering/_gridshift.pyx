from collections import Counter

cimport numpy as np
import numpy as np


cdef extern from "gridshift.h":
    void grid_cluster(int n,
                      int d,
                      int base,
                      int iterations,
                      float bandwidth,
                      int * offsets,
                      float * X_shifted,
                      int * membership,
                      int * k_num)

cdef extern from "utils.h":
    void generate_offsets_cy(int d,
                             int base,
                             int * offsets)


cdef generate_offsets_np(d,
                         base,
                         np.ndarray[np.int32_t, ndim=2, mode="c"] offsets):
    generate_offsets_cy(d,
                        base,
                        <int *> np.PyArray_DATA(offsets))

cdef shift_np(n,
              d,
              base,
              iterations,
              bandwidth,
              np.ndarray[np.int32_t, ndim=2, mode="c"] offsets,
              np.ndarray[float, ndim=2, mode="c"] X_shifted,
              np.ndarray[np.int32_t, ndim=1, mode="c"] membership,
              np.ndarray[np.int32_t, ndim=1, mode="c"] k_num):
    grid_cluster(n,
                 d,
                 base,
                 iterations,
                 bandwidth,
                 <int *> np.PyArray_DATA(offsets),
                 <float *> np.PyArray_DATA(X_shifted),
                 <int *> np.PyArray_DATA(membership),
                 <int *> np.PyArray_DATA(k_num))


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
        generate_offsets_np(d, base, offsets)
        k = np.full((1,), -1, dtype=np.int32);

        shift_np(n,
                 d,
                 base,
                 self.max_iters,
                 self.bandwidth,
                 offsets,
                 X_shifted,
                 membership,
                 k)

        cluster_centers = X_shifted[0:np.ndarray.item(k), :];
        if return_centers:
            return cluster_centers, membership
        else:
            return membership
