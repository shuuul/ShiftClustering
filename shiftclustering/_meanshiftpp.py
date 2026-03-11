"""
MeanShift++ clustering algorithm.

The C++ kernel (``meanshiftpp.h``) performs one shift iteration: bin all
points by floor-dividing coordinates by ``bandwidth``, then move each bin
to the weighted mean of its 3^d neighbors.  This module drives the
iterative loop, convergence check, and final label extraction.
"""

import numpy as np

from ._meanshiftpp_nb import _generate_offsets, _shift_cy


def meanshiftpp(X, bandwidth, threshold=1.0e-4, max_iter=300, return_centers=False):
    """
    Functional interface for MeanShift++ clustering algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data points.
    bandwidth : float
        Radius for binning points. Points are assigned to the bin
        corresponding to floor division by bandwidth.
    threshold : float, default=1.0e-4
        Stop shifting if the L2 norm between iterations is less than threshold.
    max_iter : int, default=300
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
    >>> from shiftclustering import meanshiftpp
    >>> X = np.random.rand(100, 2)
    >>> labels = meanshiftpp(X, bandwidth=0.1)
    >>> centers, labels = meanshiftpp(X, bandwidth=0.1, return_centers=True)
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    X_shifted = np.copy(X)

    base = 3
    offsets = np.full((base**d, d), -1, dtype=np.int32)
    _generate_offsets(d, base, offsets)

    iteration = 0
    while not max_iter or iteration < max_iter:
        iteration += 1
        _shift_cy(n, d, base, bandwidth, offsets, X_shifted)
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
        Convergence criterion -- stop when the L2 norm of the shift
        between consecutive iterations falls below this value.
    max_iter : int, default=300
        Maximum number of shift iterations.

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
        return meanshiftpp(
            X, self.bandwidth, self.threshold, self.iterations, return_centers
        )
