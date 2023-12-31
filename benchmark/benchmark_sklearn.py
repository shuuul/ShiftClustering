from time import time

from sklearn.cluster import MeanShift, DBSCAN
from sklearn.datasets import make_blobs

from shiftclustering import MeanShiftPP, GridShift


# Create a function to run and time a clustering algorithm
def benchmark_clustering(clustering_algo, X, labels_true):
    start_time = time()
    clustering_algo.fit_predict(X)
    end_time = time()
    return end_time - start_time


# Create a random dataset with 10,000 samples, 4 features, and 4000 centers
X, labels_true = make_blobs(n_samples=50000, n_features=4, centers=1000, random_state=42)

# Initialize the clustering algorithms
meanshift = MeanShift(bandwidth=1.0, bin_seeding=True, n_jobs=10)
meanshiftpp = MeanShiftPP(bandwidth=1.0)
gridshift = GridShift(bandwidth=1.0)
# dbscan = DBSCAN(eps=2.0)

# Benchmark the algorithms
times = {}
times['MeanShift'] = benchmark_clustering(meanshift, X, labels_true)
# times['DBSCAN'] = benchmark_clustering(dbscan, X, labels_true)
times['MeanShiftPP'] = benchmark_clustering(meanshiftpp, X, labels_true)
times['GridShift'] = benchmark_clustering(gridshift, X, labels_true)

print(times)
