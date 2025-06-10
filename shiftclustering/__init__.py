"""
ShiftClustering: Fast clustering algorithms with OpenMP parallel support

This package provides implementations of:
- MeanShiftPP: An optimized mean shift clustering algorithm
- GridShift: A grid-based clustering algorithm

Both algorithms support parallel processing via the n_jobs parameter.
"""

from ._meanshiftpp import MeanShiftPP
from ._gridshift import GridShift

__version__ = "0.1.0"
__all__ = ["MeanShiftPP", "GridShift"]
