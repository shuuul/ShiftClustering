#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3
"""
Cython wrapper for the GridShift clustering algorithm.

The C++ kernel (``gridshift.h``) performs grid-based clustering by
iteratively computing neighbour means on a binned grid and re-binning
until no membership changes occur.  Unlike MeanShiftPP, GridShift
tracks cluster membership directly on the grid rather than on individual
data points, making it faster for large datasets.
"""

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


def gridshift(X, bandwidth, threshold=1.0e-4, max_iters=300, return_centers=False):
    """
    Functional interface for GridShift clustering algorithm.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points
    bandwidth : float
        Radius for binning points. Points are assigned to the bin
        corresponding to floor division by bandwidth
    threshold : float, default=1.0e-4
        Stop shifting if the L2 norm between iterations is less than threshold
    max_iters : int, default=300
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
    >>> from shiftclustering import gridshift
    >>> X = np.random.rand(100, 2)
    >>> labels = gridshift(X, bandwidth=0.1)
    >>> centers, labels = gridshift(X, bandwidth=0.1, return_centers=True)
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
    cdef int c_max_iters = max_iters
    cdef float c_bandwidth = bandwidth
    
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


class GridShift:
    """
    Scikit-learn-style interface for GridShift clustering.

    GridShift operates on a grid representation of the data. At each
    iteration it accumulates neighbor statistics across 3^d adjacent
    bins and re-bins the shifted means. Convergence is reached when no
    bin memberships change between iterations.

    Parameters
    ----------
    bandwidth : float
        Radius for binning points. Coordinates are discretised via
        floor division by this value.
    threshold : float, default=1.0e-4
        Convergence criterion — stop when the L2 norm of the shift
        between consecutive iterations falls below this value.
    max_iters : int, default=300
        Maximum number of shift iterations.

    Attributes
    ----------
    bandwidth : float
    threshold : float
    max_iters : int

    Examples
    --------
    >>> import numpy as np
    >>> from shiftclustering import GridShift
    >>> X = np.random.rand(1000, 2).astype(np.float32)
    >>> gs = GridShift(bandwidth=0.1)
    >>> labels = gs.fit_predict(X)
    """

    def __init__(self, bandwidth, threshold=1.0e-4, max_iters=300):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.max_iters = max_iters

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
        return gridshift(X, self.bandwidth, self.threshold, self.max_iters, return_centers)
