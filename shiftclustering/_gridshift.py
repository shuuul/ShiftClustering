"""
GridShift clustering algorithm.

The C++ kernel (``gridshift.h``) performs grid-based clustering by
iteratively computing neighbour means on a binned grid and re-binning
until no membership changes occur.  Unlike MeanShiftPP, GridShift
tracks cluster membership directly on the grid rather than on individual
data points, making it faster for large datasets.
"""

import numpy as np

from ._gridshift_nb import _generate_offsets, _grid_cluster


def gridshift(X, bandwidth, threshold=1.0e-4, max_iters=300, return_centers=False):
    """
    Functional interface for GridShift clustering algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points.
    bandwidth : float
        Radius for binning points. Points are assigned to the bin
        corresponding to floor division by bandwidth.
    threshold : float, default=1.0e-4
        Convergence criterion (unused by the C++ kernel, kept for API
        consistency).
    max_iters : int, default=300
        Maximum number of iterations to run.
    return_centers : bool, default=False
        If True, return cluster centers along with labels.

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels for each point.
    centers : ndarray, shape (n_clusters, n_features), optional
        Cluster centers (only returned if return_centers=True).

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
    offsets = np.full((base**d, d), -1, dtype=np.int32)
    k = np.full((1,), -1, dtype=np.int32)

    _generate_offsets(d, base, offsets)
    _grid_cluster(n, d, base, max_iters, bandwidth, offsets, X_shifted, membership, k)

    cluster_centers = X_shifted[0 : np.ndarray.item(k), :]
    if return_centers:
        return cluster_centers, membership
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
        Convergence criterion -- stop when the L2 norm of the shift
        between consecutive iterations falls below this value.
    max_iters : int, default=300
        Maximum number of shift iterations.

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
        return gridshift(
            X, self.bandwidth, self.threshold, self.max_iters, return_centers
        )
