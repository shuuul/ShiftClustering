#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3
"""
Cython wrapper for the MeanShift++ clustering algorithm.

The C++ kernel (``meanshiftpp.h``) performs one shift iteration: bin all
points by floor-dividing coordinates by ``bandwidth``, then move each bin
to the weighted mean of its 3^d neighbors.  This module drives the
iterative loop, convergence check, and final label extraction.
"""

from collections import Counter

cimport numpy as cnp
import numpy as np

cnp.import_array()

ctypedef cnp.int32_t int32_t
ctypedef cnp.float32_t float32_t


cdef extern from "meanshiftpp.h":
    void shift_cy(int n,
                  int d,
                  int base,
                  float bandwidth,
                  int * offsets,
                  float * X_shifted) nogil


cdef extern from "utils.h":
    void generate_offsets_cy(int d,
                             int base,
                             int * offsets) nogil


cdef void generate_offsets_np(int d,
                              int base,
                              int32_t[:, ::1] offsets) nogil:
    generate_offsets_cy(d, base, <int*>&offsets[0, 0])


def meanshiftpp(X, bandwidth, threshold=1.0e-4, max_iter=300, return_centers=False):
    """
    Functional interface for MeanShift++ clustering algorithm.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points
    bandwidth : float
        Radius for binning points. Points are assigned to the bin
        corresponding to floor division by bandwidth
    threshold : float, default=1.0e-4
        Stop shifting if the L2 norm between iterations is less than threshold
    max_iter : int, default=300
        Maximum number of iterations to run
    return_centers : bool, default=False
        If True, return cluster centers along with labels
        
    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels for each point
    centers : ndarray, shape (n_clusters, n_features), optional
        Cluster centers (only returned if return_centers=True)
        
    Examples
    --------
    >>> import numpy as np
    >>> from shiftclustering import meanshiftpp
    >>> X = np.random.rand(100, 2)
    >>> labels = meanshiftpp(X, bandwidth=0.1)
    >>> centers, labels = meanshiftpp(X, bandwidth=0.1, return_centers=True)
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    X_shifted = np.copy(X)
    result = np.full(n, -1, dtype=np.int32)

    iteration = 0
    base = 3
    offsets = np.full((base ** d, d), -1, dtype=np.int32)
    
    # Create memory views for efficient access
    cdef int32_t[:, ::1] offsets_view = offsets
    cdef float32_t[:, ::1] X_shifted_view
    cdef int c_n = n
    cdef int c_d = d
    cdef int c_base = base
    cdef float c_bandwidth = bandwidth
    
    generate_offsets_np(d, base, offsets_view)

    while not max_iter or iteration < max_iter:
        iteration += 1
        X_shifted_view = X_shifted
        
        # Use sequential shift function
        with nogil:
            shift_cy(c_n, c_d, c_base, c_bandwidth, 
                    <int*>&offsets_view[0, 0], <float*>&X_shifted_view[0, 0])

        if np.linalg.norm(np.subtract(X, X_shifted)) <= threshold:
            break
        X = np.copy(X_shifted)

    cluster_centers, result = np.unique(X_shifted, return_inverse=True, axis=0)
    if return_centers:
        return cluster_centers, result

    return result


class MeanShiftPP:
    """
    Scikit-learn-style interface for MeanShift++ clustering.

    The algorithm iterates between two steps until convergence:
    1. Bin all points by floor-dividing coordinates by ``bandwidth``.
    2. Shift each bin to the weighted mean of its 3^d neighbors.

    Points sharing the same final bin are assigned the same cluster label.

    Parameters
    ----------
    bandwidth : float
        Radius for binning points. Coordinates are discretised via
        floor division by this value.
    threshold : float, default=1.0e-4
        Convergence criterion — stop when the L2 norm of the shift
        between consecutive iterations falls below this value.
    max_iter : int, default=300
        Maximum number of shift iterations.

    Attributes
    ----------
    bandwidth : float
    threshold : float
    iterations : int

    Examples
    --------
    >>> import numpy as np
    >>> from shiftclustering import MeanShiftPP
    >>> X = np.random.rand(500, 2).astype(np.float32)
    >>> ms = MeanShiftPP(bandwidth=0.1)
    >>> labels = ms.fit_predict(X)
    """

    def __init__(self, bandwidth, threshold=1.0e-4, max_iter=300):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.iterations = max_iter

    def fit_predict(self, X, return_centers=False):
        """
        Cluster the data and return labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data points. Will be cast to float32 C-contiguous.
        return_centers : bool, default=False
            If True, return ``(centers, labels)`` instead of just labels.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster label for each sample.
        centers : ndarray, shape (n_clusters, n_features), optional
            Only returned when ``return_centers=True``.
        """
        return meanshiftpp(X, self.bandwidth, self.threshold, self.iterations, return_centers)
