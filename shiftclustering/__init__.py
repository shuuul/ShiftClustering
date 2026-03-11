"""
ShiftClustering: Fast clustering algorithms with C++/nanobind and OpenMP support.

Provides three algorithms with both functional and scikit-learn-style class APIs:

- :func:`meanshiftpp` / :class:`MeanShiftPP`
    Optimized mean shift clustering using grid-based binning and neighbor shifting.
- :func:`localshift` / :class:`LocalShift`
    3D point cloud optimization for cryo-EM density maps via Gaussian kernel
    weighted shifting with OpenMP parallelism.
- :func:`gridshift` / :class:`GridShift`
    Grid-based iterative clustering that operates directly on binned representations.

All algorithms accept float32 arrays and are implemented as nanobind wrappers
around C++ kernels for maximum throughput.
"""

from ._meanshiftpp import MeanShiftPP, meanshiftpp
from ._localshift import LocalShift, localshift
from ._gridshift import GridShift, gridshift

__version__ = "0.2.0"

__all__ = [
    "MeanShiftPP", "LocalShift", "GridShift",
    "meanshiftpp", "localshift", "gridshift",
]
