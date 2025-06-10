"""
Official benchmark comparing MeanShift implementations.

This benchmark compares three clustering algorithms:
- sklearn.cluster.MeanShift (baseline)
- shiftclustering.MeanShiftPP (our optimized implementation)
- shiftclustering.GridShift (our grid-based implementation)

The sample size is kept moderate since sklearn MeanShift is computationally expensive.
"""

import statistics
from time import time
from typing import Any, Dict, Protocol

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from shiftclustering import MeanShiftPP, GridShift


class ClusteringAlgorithm(Protocol):
    """Protocol for clustering algorithms."""
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the algorithm and predict cluster labels."""
        ...


def benchmark_clustering(
    clustering_algo: ClusteringAlgorithm,
    X: np.ndarray,
    n_runs: int = 3
) -> Dict[str, Any]:
    """
    Benchmark a clustering algorithm with multiple runs.
    
    Args:
        clustering_algo: The clustering algorithm to benchmark
        X: Input data
        n_runs: Number of runs for statistical stability
        
    Returns:
        Dictionary with timing and quality metrics
    """
    times = []
    labels_list = []
    
    for _ in range(n_runs):
        start_time = time()
        labels = clustering_algo.fit_predict(X)
        end_time = time()
        
        times.append(end_time - start_time)
        labels_list.append(labels)
    
    # Calculate timing statistics
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    
    # Use the first run's labels for quality metrics
    labels = labels_list[0]
    n_clusters = len(set(labels))
    
    # Calculate silhouette score (quality metric)
    try:
        sil_score = silhouette_score(X, labels) if n_clusters > 1 else 0.0
    except ValueError:
        sil_score = 0.0
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'n_clusters': n_clusters,
        'silhouette_score': sil_score,
        'all_times': times
    }


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "="*80)
    print("CLUSTERING ALGORITHM BENCHMARK RESULTS")
    print("="*80)
    
    print(f"{'Algorithm':<15} {'Time (s)':<12} {'±Std':<8} {'Clusters':<10} {'Silhouette':<12}")
    print("-" * 80)
    
    for algo_name, metrics in results.items():
        print(f"{algo_name:<15} "
              f"{metrics['mean_time']:<12.4f} "
              f"±{metrics['std_time']:<7.4f} "
              f"{metrics['n_clusters']:<10} "
              f"{metrics['silhouette_score']:<12.4f}")
    
    # Calculate speedup relative to sklearn MeanShift
    if 'MeanShift' in results:
        baseline_time = results['MeanShift']['mean_time']
        print("\n" + "-" * 80)
        print("SPEEDUP RELATIVE TO SKLEARN MEANSHIFT:")
        print("-" * 80)
        
        for algo_name, metrics in results.items():
            if algo_name != 'MeanShift':
                speedup = baseline_time / metrics['mean_time']
                print(f"{algo_name:<15} {speedup:<12.2f}x faster")


def main():
    """Run the main benchmark."""
    # Configuration
    n_samples = 3000  # Moderate size due to sklearn MeanShift being slow
    n_features = 3
    n_centers = 20
    bandwidth = 1.5
    n_runs = 3
    random_state = 42
    
    print("ShiftClustering Algorithm Benchmark")
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_centers} centers")
    print(f"Bandwidth: {bandwidth}, Runs per algorithm: {n_runs}")
    print(f"Random state: {random_state}")
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    X, y_true = make_blobs(
        n_samples=n_samples, 
        n_features=n_features, 
        centers=n_centers, 
        random_state=random_state,
        cluster_std=1.0
    )
    
    # Initialize algorithms
    print("Initializing clustering algorithms...")
    algorithms = {
        'MeanShift': MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=1),
        'MeanShiftPP': MeanShiftPP(bandwidth=bandwidth, n_jobs=1),
        'GridShift': GridShift(bandwidth=bandwidth, n_jobs=1)
    }
    
    # Run benchmarks
    results = {}
    for algo_name, algo in algorithms.items():
        print(f"\nBenchmarking {algo_name}...")
        results[algo_name] = benchmark_clustering(algo, X, n_runs)
    
    # Print results
    print_results(results)
    
    # Additional analysis
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    fastest_algo = min(results.keys(), key=lambda k: results[k]['mean_time'])
    highest_quality = max(results.keys(), key=lambda k: results[k]['silhouette_score'])
    
    print(f"Fastest algorithm: {fastest_algo} ({results[fastest_algo]['mean_time']:.4f}s)")
    print(f"Highest quality: {highest_quality} (silhouette: {results[highest_quality]['silhouette_score']:.4f})")
    
    print("\nDataset statistics:")
    print(f"- True number of clusters: {n_centers}")
    print(f"- Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"- Data shape: {X.shape}")


if __name__ == "__main__":
    main()
