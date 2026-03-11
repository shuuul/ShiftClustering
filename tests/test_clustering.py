"""Tests for all ShiftClustering algorithms."""

import numpy as np
import pytest

from shiftclustering import (
    GridShift,
    LocalShift,
    MeanShiftPP,
    gridshift,
    localshift,
    meanshiftpp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blobs(n_per_cluster=50, centers=None, rng=None):
    """Create well-separated 2D blobs for deterministic clustering tests."""
    if rng is None:
        rng = np.random.default_rng(42)
    if centers is None:
        centers = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    d = centers.shape[1]
    parts = []
    labels = []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=0.3, size=(n_per_cluster, d)).astype(np.float32)
        parts.append(pts)
        labels.append(np.full(n_per_cluster, i, dtype=np.int32))
    return np.vstack(parts), np.concatenate(labels)


def make_density_peak(shape=(30, 30, 30), center=None):
    """Create a 3D Gaussian density map with a single peak."""
    if center is None:
        center = np.array([s // 2 for s in shape], dtype=np.float32)
    grid = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    d2 = sum((grid[i] - center[i]) ** 2 for i in range(3))
    density = np.exp(-0.1 * d2).astype(np.float32)
    return density, center


# ---------------------------------------------------------------------------
# MeanShiftPP
# ---------------------------------------------------------------------------

class TestMeanShiftPP:
    def test_functional_two_blobs(self):
        X, true_labels = make_blobs()
        labels = meanshiftpp(X, bandwidth=1.0)
        assert labels.shape == (len(X),)
        n_clusters = len(np.unique(labels))
        assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}"

    def test_functional_return_centers(self):
        X, _ = make_blobs()
        centers, labels = meanshiftpp(X, bandwidth=1.0, return_centers=True)
        assert centers.ndim == 2
        assert centers.shape[1] == X.shape[1]
        assert len(centers) == len(np.unique(labels))

    def test_class_interface(self):
        X, _ = make_blobs()
        ms = MeanShiftPP(bandwidth=1.0)
        labels = ms.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(np.unique(labels)) == 2

    def test_class_return_centers(self):
        X, _ = make_blobs()
        ms = MeanShiftPP(bandwidth=1.0)
        centers, labels = ms.fit_predict(X, return_centers=True)
        assert centers.shape[1] == 2

    def test_single_cluster(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=0, scale=0.1, size=(30, 2)).astype(np.float32)
        labels = meanshiftpp(X, bandwidth=1.0)
        assert len(np.unique(labels)) == 1

    def test_dtype_coercion(self):
        X_f64 = np.random.default_rng(0).normal(size=(40, 2))
        labels = meanshiftpp(X_f64, bandwidth=1.0)
        assert labels.dtype == np.intp or np.issubdtype(labels.dtype, np.integer)

    def test_higher_dim(self):
        rng = np.random.default_rng(7)
        c1 = np.zeros(3, dtype=np.float32)
        c2 = np.full(3, 10.0, dtype=np.float32)
        X, _ = make_blobs(centers=np.stack([c1, c2]))
        labels = meanshiftpp(X, bandwidth=1.0)
        assert len(np.unique(labels)) == 2


# ---------------------------------------------------------------------------
# GridShift
# ---------------------------------------------------------------------------

class TestGridShift:
    def test_functional_two_blobs(self):
        X, _ = make_blobs()
        labels = gridshift(X, bandwidth=1.0)
        assert labels.shape == (len(X),)
        n_clusters = len(np.unique(labels))
        assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}"

    def test_functional_return_centers(self):
        X, _ = make_blobs()
        centers, labels = gridshift(X, bandwidth=1.0, return_centers=True)
        assert centers.ndim == 2
        assert centers.shape[1] == X.shape[1]

    def test_class_interface(self):
        X, _ = make_blobs()
        gs = GridShift(bandwidth=1.0)
        labels = gs.fit_predict(X)
        assert labels.shape == (len(X),)
        assert len(np.unique(labels)) == 2

    def test_class_return_centers(self):
        X, _ = make_blobs()
        gs = GridShift(bandwidth=1.0)
        centers, labels = gs.fit_predict(X, return_centers=True)
        assert centers.shape[1] == 2

    def test_single_cluster(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=0, scale=0.1, size=(30, 2)).astype(np.float32)
        labels = gridshift(X, bandwidth=1.0)
        assert len(np.unique(labels)) == 1


# ---------------------------------------------------------------------------
# LocalShift
# ---------------------------------------------------------------------------

class TestLocalShift:
    def test_functional_shifts_toward_peak(self):
        density, center = make_density_peak()
        rng = np.random.default_rng(99)
        pts = (center + rng.uniform(-3, 3, size=(20, 3))).astype(np.float32)
        shifted = localshift(pts, density, fmaxd=5.0, fsiv=0.5, n_steps=50)
        assert shifted.shape == pts.shape
        dists_before = np.linalg.norm(pts - center, axis=1)
        dists_after = np.linalg.norm(shifted - center, axis=1)
        assert np.mean(dists_after) < np.mean(dists_before)

    def test_class_interface(self):
        density, center = make_density_peak()
        rng = np.random.default_rng(99)
        pts = (center + rng.uniform(-3, 3, size=(10, 3))).astype(np.float32)
        ls = LocalShift(fmaxd=5.0, fsiv=0.5, n_steps=50)
        shifted = ls.fit_predict(pts, density)
        assert shifted.shape == pts.shape

    def test_single_thread(self):
        density, center = make_density_peak()
        pts = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        shifted = localshift(pts, density, fmaxd=5.0, fsiv=0.5, n_steps=50, n_jobs=1)
        assert shifted.shape == (1, 3)
        assert np.linalg.norm(shifted[0] - center) < np.linalg.norm(pts[0] - center)

    def test_input_validation_2d(self):
        density, _ = make_density_peak()
        with pytest.raises(ValueError, match="2D array"):
            localshift(np.array([1, 2, 3], dtype=np.float32), density, 5.0, 0.5)

    def test_input_validation_cols(self):
        density, _ = make_density_peak()
        with pytest.raises(ValueError, match="2D array"):
            localshift(np.zeros((5, 2), dtype=np.float32), density, 5.0, 0.5)

    def test_input_validation_ref(self):
        pts = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="3D array"):
            localshift(pts, np.zeros((10, 10), dtype=np.float32), 5.0, 0.5)

    def test_does_not_mutate_input(self):
        density, center = make_density_peak()
        pts = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        original = pts.copy()
        localshift(pts, density, fmaxd=5.0, fsiv=0.5, n_steps=50)
        np.testing.assert_array_equal(pts, original)

    def test_dtype_coercion(self):
        density, center = make_density_peak()
        pts_f64 = np.array([[15.0, 15.0, 15.0]])
        shifted = localshift(pts_f64, density, fmaxd=5.0, fsiv=0.5, n_steps=10)
        assert shifted.dtype == np.float32
