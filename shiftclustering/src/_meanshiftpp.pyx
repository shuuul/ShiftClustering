#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3

from collections import Counter

cimport numpy as cnp
import numpy as np
from cython.parallel cimport parallel, prange

# Try to import OpenMP, fallback if not available
cdef bint OPENMP_ENABLED = True
try:
    from openmp cimport omp_get_max_threads, omp_set_num_threads
except ImportError:
    OPENMP_ENABLED = False

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


cdef inline int get_max_threads() nogil:
    """Get maximum number of threads, fallback to 1 if OpenMP not available"""
    if OPENMP_ENABLED:
        return omp_get_max_threads()
    else:
        return 1


cdef void generate_offsets_np(int d,
                              int base,
                              int32_t[:, ::1] offsets) nogil:
    generate_offsets_cy(d, base, <int*>&offsets[0, 0])


cdef void shift_np_parallel(int n,
                           int d,
                           int base,
                           float bandwidth,
                           int32_t[:, ::1] offsets,
                           float32_t[:, ::1] X_shifted,
                           int n_jobs) nogil:
    """Parallel wrapper for shift_cy using chunking strategy"""
    
    cdef int num_threads = n_jobs
    if n_jobs == -1:
        num_threads = get_max_threads()
    elif n_jobs <= 0:
        num_threads = 1
    
    # For small datasets or no OpenMP, use single-threaded to avoid overhead
    if n < 1000 or num_threads == 1 or not OPENMP_ENABLED:
        shift_cy(n, d, base, bandwidth, <int*>&offsets[0, 0], <float*>&X_shifted[0, 0])
        return
    
    # Calculate chunk size for parallel processing
    cdef int chunk_size = max(100, n // num_threads)
    cdef int i, start, end
    
    # Process chunks in parallel
    if OPENMP_ENABLED:
        with parallel(num_threads=num_threads):
            for i in prange(0, n, chunk_size, schedule='static'):
                start = i
                end = min(i + chunk_size, n)
                
                # Process this chunk
                shift_cy(end - start, d, base, bandwidth, 
                        <int*>&offsets[0, 0], 
                        <float*>&X_shifted[start, 0])
    else:
        # Fallback to sequential processing
        shift_cy(n, d, base, bandwidth, <int*>&offsets[0, 0], <float*>&X_shifted[0, 0])


class MeanShiftPP:
    """
    Parameters
    ----------

    bandwidth: Radius for binning points. Points are assigned to the bin
               corresponding to floor division by bandwidth

    threshold: Stop shifting if the L2 norm between max_iters is less than
               threshold

    max_iter: Maximum number of max_iters to run
    
    n_jobs: Number of parallel threads to use. If -1, use all available cores.
            If 1 (default), disable parallelism.
            Note: Parallel processing requires OpenMP support.

    """

    def __init__(self, bandwidth, threshold=1.0e-4, max_iter=300, n_jobs=1):
        self.bandwidth = bandwidth
        self.threshold = threshold
        self.iterations = max_iter
        self.n_jobs = n_jobs
        
        # Warn if n_jobs > 1 but OpenMP is not available
        if n_jobs != 1 and not OPENMP_ENABLED:
            import warnings
            warnings.warn("OpenMP not available. Parallel processing disabled. "
                         "Install OpenMP to enable parallel processing.", 
                         RuntimeWarning)

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
        cdef float c_bandwidth = self.bandwidth
        cdef int c_n_jobs = self.n_jobs
        
        generate_offsets_np(d, base, offsets_view)

        while not self.iterations or iteration < self.iterations:
            iteration += 1
            X_shifted_view = X_shifted
            
            # Use parallel shift function
            with nogil:
                shift_np_parallel(c_n, c_d, c_base, c_bandwidth, 
                                offsets_view, X_shifted_view, c_n_jobs)

            if np.linalg.norm(np.subtract(X, X_shifted)) <= self.threshold:
                break
            X = np.copy(X_shifted)

        cluster_centers, result = np.unique(X_shifted, return_inverse=True, axis=0)
        if return_centers:
            return cluster_centers, result

        return result
