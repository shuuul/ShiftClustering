from time import time

from sklearn.cluster import MeanShift, DBSCAN
from sklearn.datasets import make_blobs

from shiftclustering import MeanShiftPP, GridShiftPP


# Create a function to run and time a clustering algorithm
def benchmark_clustering(clustering_algo, X):
    start_time = time()
    Y_pred = clustering_algo.fit_predict(X)
    end_time = time()
    return end_time - start_time


# Create a random dataset with 10,000 samples, 2 features, and 3 centers
X, _ = make_blobs(n_samples=10000, n_features=2, centers=3, random_state=42)

# Initialize the clustering algorithms
meanshift = MeanShift(bandwidth=0.3)
meanshiftpp = MeanShiftPP(bandwidth=0.3)
gridshiftpp = GridShiftPP(bandwidth=0.3)
dbscan = DBSCAN(eps=0.3)

# Benchmark the algorithms
times = {}
times['MeanShift'] = benchmark_clustering(meanshift, X)
times['DBSCAN'] = benchmark_clustering(dbscan, X)
times['MeanShiftPP'] = benchmark_clustering(meanshiftpp, X)
times['GridShiftPP'] = benchmark_clustering(gridshiftpp, X)

print(times)
