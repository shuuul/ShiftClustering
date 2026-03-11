"""
LocalShift 3D point-cloud optimisation algorithm.

The C++ kernel (``localshift.h``) shifts a single point towards the local
maximum of a 3D reference density map using Gaussian kernel weighting.
The nanobind wrapper parallelises the per-point calls across all available
cores via OpenMP.

Reference
---------
Terashi, G. and Kihara, D. "De novo main-chain modeling for EM maps
using MAINMAST." *Nature Communications* 9, 1618 (2018).
"""

import os

import numpy as np

from ._localshift_nb import _localshift_parallel


def localshift(point_cd, reference, fmaxd, fsiv, n_steps=100, tol=1e-6, n_jobs=-1):
    """
    Local shift algorithm for 3D point cloud optimization in cryo-EM context.

    This algorithm iteratively shifts points in 3D space towards local maxima
    in a reference density map using a Gaussian kernel weighting scheme.
    Commonly used in cryo-EM structure refinement to find optimal atomic
    positions.

    Parameters
    ----------
    point_cd : array-like, shape (n_points, 3)
        Initial 3D coordinates of points to be shifted.
    reference : array-like, shape (D, H, W)
        3D reference density map (e.g., cryo-EM map).
    fmaxd : float
        Maximum distance for neighbor search around each point.
    fsiv : float
        Kernel parameter for Gaussian weighting (higher values = more
        localized).
    n_steps : int, default=100
        Maximum number of iterations per point.
    tol : float, default=1e-6
        Tolerance for convergence (squared distance threshold).
    n_jobs : int, default=-1
        Number of parallel threads to use (-1 = use all available cores).

    Returns
    -------
    shifted_points : ndarray, shape (n_points, 3)
        Optimized 3D coordinates after local shift.

    Examples
    --------
    >>> import numpy as np
    >>> from shiftclustering import localshift
    >>> atoms = np.random.rand(50, 3) * 100
    >>> density_map = np.random.rand(100, 100, 100)
    >>> optimized_atoms = localshift(atoms, density_map, fmaxd=5.0, fsiv=0.1)
    """
    point_cd = np.ascontiguousarray(point_cd, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)

    if point_cd.ndim != 2 or point_cd.shape[1] != 3:
        raise ValueError("point_cd must be a 2D array with shape (n_points, 3)")
    if reference.ndim != 3:
        raise ValueError("reference must be a 3D array")

    shifted_points = np.copy(point_cd)

    max_threads = os.cpu_count() or 4
    if n_jobs == -1:
        num_threads = max_threads
    elif n_jobs > 0:
        num_threads = min(n_jobs, max_threads)
    else:
        num_threads = 1

    _localshift_parallel(
        shifted_points, reference,
        float(fmaxd), float(fsiv), int(n_steps), float(tol), num_threads,
    )

    return shifted_points


class LocalShift:
    """
    Local shift algorithm for 3D point cloud optimization in cryo-EM context.

    This algorithm iteratively shifts points in 3D space towards local maxima
    in a reference density map using a Gaussian kernel weighting scheme.

    Parameters
    ----------
    fmaxd : float
        Maximum distance for neighbor search around each point.
    fsiv : float
        Kernel parameter for Gaussian weighting (higher values = more
        localized).
    n_steps : int, default=100
        Maximum number of iterations per point.
    tol : float, default=1e-6
        Tolerance for convergence (squared distance threshold).
    n_jobs : int, default=-1
        Number of parallel threads to use (-1 = use all available cores).
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
            Initial 3D coordinates of points to be shifted.
        reference : array-like, shape (D, H, W)
            3D reference density map (e.g., cryo-EM map).

        Returns
        -------
        shifted_points : ndarray, shape (n_points, 3)
            Optimized 3D coordinates after local shift.
        """
        return localshift(
            point_cd, reference, self.fmaxd, self.fsiv,
            self.n_steps, self.tol, self.n_jobs,
        )
