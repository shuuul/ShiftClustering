"""
ShiftClustering: Fast clustering algorithms with OpenMP parallel support

This package provides implementations of:
- MeanShiftPP: An optimized mean shift clustering algorithm
- LocalShift: A local shift algorithm for 3D point cloud optimization (cryo-EM)
- GridShift: A grid-based clustering algorithm

The algorithms support efficient processing and are optimized for performance.
"""

# Import class interfaces
from ._meanshiftpp import MeanShiftPP, meanshiftpp
from ._localshift import LocalShift, localshift
from ._gridshift import GridShift, gridshift

__version__ = "0.1.0"

__all__ = [
    # Class interfaces
    "MeanShiftPP", "LocalShift", "GridShift",
    # Functional interfaces
    "meanshiftpp", "localshift", "gridshift"
]

