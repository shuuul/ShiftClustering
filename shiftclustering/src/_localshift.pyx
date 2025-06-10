#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3

cimport numpy as cnp
import numpy as np
from cython.parallel cimport parallel, prange

cnp.import_array()

ctypedef cnp.int32_t int32_t
ctypedef cnp.float32_t float32_t


cdef extern from "localshift.h":
    void localshift_single_cy(float * point_pos,
                              int n_steps,
                              float fmaxd,
                              float fsiv,
                              float tol,
                              int ref_shape_0,
                              int ref_shape_1,
                              int ref_shape_2,
                              float * reference) nogil


def localshift(point_cd, reference, fmaxd, fsiv, n_steps=100, tol=1e-6, n_jobs=-1):
    """
    Local shift algorithm for 3D point cloud optimization in cryo-EM context.
    
    This algorithm iteratively shifts points in 3D space towards local maxima
    in a reference density map using a Gaussian kernel weighting scheme.
    Commonly used in cryo-EM structure refinement to find optimal atomic positions.
    
    Parameters
    ----------
    point_cd : array-like, shape (n_points, 3)
        Initial 3D coordinates of points to be shifted
    reference : array-like, shape (D, H, W)
        3D reference density map (e.g., cryo-EM map)
    fmaxd : float
        Maximum distance for neighbor search around each point
    fsiv : float
        Kernel parameter for Gaussian weighting (higher values = more localized)
    n_steps : int, default=100
        Maximum number of iterations per point
    tol : float, default=1e-6
        Tolerance for convergence (squared distance threshold)
    n_jobs : int, default=-1
        Number of parallel threads to use (-1 = use all available cores)
        
    Returns
    -------
    shifted_points : ndarray, shape (n_points, 3)
        Optimized 3D coordinates after local shift
        
    Examples
    --------
    >>> import numpy as np
    >>> from shiftclustering import localshift
    >>> # Initial atom positions
    >>> atoms = np.random.rand(50, 3) * 100
    >>> # Density map (synthetic example)
    >>> density_map = np.random.rand(100, 100, 100)
    >>> # Optimize positions
    >>> optimized_atoms = localshift(atoms, density_map, fmaxd=5.0, fsiv=0.1)
    """
    # Ensure input arrays are contiguous and correct dtype
    point_cd = np.ascontiguousarray(point_cd, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    
    if point_cd.ndim != 2 or point_cd.shape[1] != 3:
        raise ValueError("point_cd must be a 2D array with shape (n_points, 3)")
        
    if reference.ndim != 3:
        raise ValueError("reference must be a 3D array")
    
    # Make a copy to avoid modifying input
    shifted_points = np.copy(point_cd)
    
    cnt = shifted_points.shape[0]
    ref_shape = reference.shape
    
    # Create memory views for efficient access
    cdef float32_t[:, ::1] points_view = shifted_points
    cdef float32_t[:, :, ::1] ref_view = reference
    
    cdef int c_cnt = cnt
    cdef int c_n_steps = n_steps
    cdef float c_fmaxd = fmaxd
    cdef float c_fsiv = fsiv
    cdef float c_tol = tol
    cdef int c_ref_shape_0 = ref_shape[0]
    cdef int c_ref_shape_1 = ref_shape[1]
    cdef int c_ref_shape_2 = ref_shape[2]
    
    # Configure number of threads
    cdef int num_threads
    import os
    if n_jobs == -1:
        # Use all available cores
        num_threads = os.cpu_count() or 4
    elif n_jobs > 0:
        num_threads = min(n_jobs, os.cpu_count() or 4)
    else:
        num_threads = 1
    
    # Process each point in parallel using prange
    cdef int i
    with nogil, parallel(num_threads=num_threads):
        for i in prange(c_cnt, schedule='dynamic'):
            localshift_single_cy(<float*>&points_view[i, 0], c_n_steps, c_fmaxd, c_fsiv, c_tol,
                                 c_ref_shape_0, c_ref_shape_1, c_ref_shape_2,
                                 <float*>&ref_view[0, 0, 0])
    
    return shifted_points


class LocalShift:
    """
    Local shift algorithm for 3D point cloud optimization in cryo-EM context.
    
    This algorithm iteratively shifts points in 3D space towards local maxima
    in a reference density map using a Gaussian kernel weighting scheme.
    
    Parameters
    ----------
    fmaxd : float
        Maximum distance for neighbor search around each point
    fsiv : float
        Kernel parameter for Gaussian weighting (higher values = more localized)
    n_steps : int, default=100
        Maximum number of iterations per point
    tol : float, default=1e-6
        Tolerance for convergence (squared distance threshold)
    n_jobs : int, default=-1
        Number of parallel threads to use (-1 = use all available cores)
    """
    
    def __init__(self, fmaxd, fsiv, n_steps=100, tol=1e-6, n_jobs=-1):
        self.fmaxd = float(fmaxd)
        self.fsiv = float(fsiv)
        self.n_steps = int(n_steps)
        self.tol = float(tol)
        self.n_jobs = int(n_jobs)
    
    def fit_predict(self, point_cd, reference):
        """
        Apply local shift algorithm to optimize point positions.
        
        Parameters
        ----------
        point_cd : array-like, shape (n_points, 3)
            Initial 3D coordinates of points to be shifted
        reference : array-like, shape (D, H, W)
            3D reference density map (e.g., cryo-EM map)
            
        Returns
        -------
        shifted_points : ndarray, shape (n_points, 3)
            Optimized 3D coordinates after local shift
        """
        return localshift(point_cd, reference, self.fmaxd, self.fsiv, 
                         self.n_steps, self.tol, self.n_jobs)
