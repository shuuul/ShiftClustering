# ShiftClustering

A Python package for MeanShift clustering and its variants MeanShift++, GridShift. This package is for learning and it not under development.

## Installation

```bash
pip install -e .
```

## Usage
```python
from sklearn.datasets import make_blobs
from shiftclustering import MeanShiftPP, GridShift

X, labels_true = make_blobs(n_samples=50000, n_features=4, centers=1000, random_state=42)

meanshiftpp = MeanShiftPP(bandwidth=1.0)
gridshift = GridShift(bandwidth=1.0)

labels_meanshiftpp = meanshiftpp.fit_predict(X)
labels_gridshift = gridshift.fit_predict(X)
```

## Acknowledgements

This repo is based on the following repository

- [meanshiftpp](https://github.com/jenniferjang/meanshiftpp)
- [GridShift](https://github.com/abhisheka456/GridShift)

Please cite their papers if you use this package.