"""
Parallel processing benchmark for ShiftClustering algorithms.

This benchmark tests the scalability of MeanShiftPP and GridShift algorithms
with different numbers of parallel jobs (n_jobs) using large datasets.

Only our custom algorithms are tested since sklearn MeanShift doesn't support
efficient parallel processing for large datasets.
"""

import multiprocessing
import statistics
from time import time
from typing import Any, Dict, List, Protocol

import numpy as np
from sklearn.datasets import make_blobs

from shiftclustering import MeanShiftPP, GridShift


class ParallelClusteringAlgorithm(Protocol):
    """Protocol for clustering algorithms with n_jobs support."""
    def __init__(self, bandwidth: float, n_jobs: int = 1, **kwargs):
        """Initialize with bandwidth and n_jobs."""
        ...
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit the algorithm and predict cluster labels."""
        ...


def benchmark_parallel_clustering(
    algo_class: type,
    X: np.ndarray,
    bandwidth: float,
    n_jobs_list: List[int],
    n_runs: int = 3,
    **kwargs
) -> Dict[int, Dict[str, Any]]:
    """
    Benchmark a clustering algorithm with different n_jobs values.
    
    Args:
        algo_class: The clustering algorithm class
        X: Input data
        bandwidth: Bandwidth parameter for clustering
        n_jobs_list: List of n_jobs values to test
        n_runs: Number of runs per n_jobs value for statistical stability
        **kwargs: Additional arguments for the algorithm
        
    Returns:
        Dictionary mapping n_jobs to performance metrics
    """
    results = {}
    
    for n_jobs in n_jobs_list:
        print(f"  Testing n_jobs={n_jobs}...")
        times = []
        labels_list = []
        
        for run in range(n_runs):
            # Create fresh algorithm instance for each run
            algo = algo_class(bandwidth=bandwidth, n_jobs=n_jobs, **kwargs)
            
            start_time = time()
            labels = algo.fit_predict(X)
            end_time = time()
            
            times.append(end_time - start_time)
            labels_list.append(labels)
            
            print(f"    Run {run+1}/{n_runs}: {end_time - start_time:.4f}s")
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        n_clusters = len(set(labels_list[0]))
        
        results[n_jobs] = {
            'mean_time': mean_time,
            'std_time': std_time,
            'n_clusters': n_clusters,
            'all_times': times
        }
    
    return results


def calculate_speedup_efficiency(results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Calculate speedup and efficiency metrics relative to single-threaded performance."""
    baseline_time = results[1]['mean_time']  # Single-threaded baseline
    
    speedup_data = {}
    for n_jobs, metrics in results.items():
        speedup = baseline_time / metrics['mean_time']
        efficiency = speedup / n_jobs if n_jobs > 0 else 0.0
        
        speedup_data[n_jobs] = {
            'speedup': speedup,
            'efficiency': efficiency
        }
    
    return speedup_data


def print_parallel_results(
    algo_name: str, 
    results: Dict[int, Dict[str, Any]], 
    speedup_data: Dict[int, Dict[str, float]]
) -> None:
    """Print parallel benchmark results in a formatted table."""
    print(f"\n{'='*90}")
    print(f"{algo_name.upper()} PARALLEL PERFORMANCE RESULTS")
    print(f"{'='*90}")
    
    print(f"{'n_jobs':<8} {'Time (s)':<12} {'±Std':<8} {'Clusters':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 90)
    
    for n_jobs in sorted(results.keys()):
        metrics = results[n_jobs]
        speedup_info = speedup_data[n_jobs]
        
        print(f"{n_jobs:<8} "
              f"{metrics['mean_time']:<12.4f} "
              f"±{metrics['std_time']:<7.4f} "
              f"{metrics['n_clusters']:<10} "
              f"{speedup_info['speedup']:<10.2f} "
              f"{speedup_info['efficiency']:<12.3f}")


def print_comparison_summary(
    meanshiftpp_results: Dict[int, Dict[str, Any]],
    gridshift_results: Dict[int, Dict[str, Any]]
) -> None:
    """Print a comparison summary between algorithms."""
    print(f"\n{'='*100}")
    print("ALGORITHM COMPARISON AT DIFFERENT n_jobs")
    print(f"{'='*100}")
    
    print(f"{'n_jobs':<8} {'MeanShiftPP (s)':<16} {'GridShift (s)':<16} {'GridShift Speedup':<18}")
    print("-" * 100)
    
    for n_jobs in sorted(meanshiftpp_results.keys()):
        ms_time = meanshiftpp_results[n_jobs]['mean_time']
        gs_time = gridshift_results[n_jobs]['mean_time']
        relative_speedup = ms_time / gs_time
        
        print(f"{n_jobs:<8} "
              f"{ms_time:<16.4f} "
              f"{gs_time:<16.4f} "
              f"{relative_speedup:<18.2f}x")


def main():
    """Run the parallel processing benchmark."""
    # Configuration
    n_samples = 50000  # Large dataset to see parallel benefits
    n_features = 4
    n_centers = 50
    bandwidth = 2.0
    n_runs = 3
    random_state = 42
    
    # Determine available CPU cores
    max_cores = multiprocessing.cpu_count()
    n_jobs_list = [1, 2, 4, max_cores, -1]  # -1 means use all cores
    
    print("ShiftClustering Parallel Processing Benchmark")
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_centers} centers")
    print(f"Bandwidth: {bandwidth}, Runs per configuration: {n_runs}")
    print(f"Available CPU cores: {max_cores}")
    print(f"Testing n_jobs values: {n_jobs_list}")
    print(f"Random state: {random_state}")
    
    # Generate large synthetic dataset
    print(f"\nGenerating large synthetic dataset ({n_samples:,} samples)...")
    X, y_true = make_blobs(
        n_samples=n_samples, 
        n_features=n_features, 
        centers=n_centers, 
        random_state=random_state,
        cluster_std=1.5
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Memory usage: ~{X.nbytes / 1024**2:.1f} MB")
    
    # Benchmark MeanShiftPP
    print(f"\nBenchmarking MeanShiftPP with different n_jobs...")
    meanshiftpp_results = benchmark_parallel_clustering(
        MeanShiftPP, X, bandwidth, n_jobs_list, n_runs
    )
    
    # Benchmark GridShift
    print(f"\nBenchmarking GridShift with different n_jobs...")
    gridshift_results = benchmark_parallel_clustering(
        GridShift, X, bandwidth, n_jobs_list, n_runs
    )
    
    # Calculate speedup and efficiency
    meanshiftpp_speedup = calculate_speedup_efficiency(meanshiftpp_results)
    gridshift_speedup = calculate_speedup_efficiency(gridshift_results)
    
    # Print results
    print_parallel_results("MeanShiftPP", meanshiftpp_results, meanshiftpp_speedup)
    print_parallel_results("GridShift", gridshift_results, gridshift_speedup)
    
    # Print comparison
    print_comparison_summary(meanshiftpp_results, gridshift_results)
    
    # Summary analysis
    print(f"\n{'='*100}")
    print("PERFORMANCE ANALYSIS SUMMARY")
    print(f"{'='*100}")
    
    # Best speedup analysis
    best_ms_speedup = max(meanshiftpp_speedup.values(), key=lambda x: x['speedup'])
    best_gs_speedup = max(gridshift_speedup.values(), key=lambda x: x['speedup'])
    
    best_ms_njobs = max(meanshiftpp_speedup.keys(), key=lambda k: meanshiftpp_speedup[k]['speedup'])
    best_gs_njobs = max(gridshift_speedup.keys(), key=lambda k: gridshift_speedup[k]['speedup'])
    
    print(f"Best MeanShiftPP speedup: {best_ms_speedup['speedup']:.2f}x at n_jobs={best_ms_njobs}")
    print(f"Best GridShift speedup: {best_gs_speedup['speedup']:.2f}x at n_jobs={best_gs_njobs}")
    
    # Efficiency analysis
    print(f"\nParallel Efficiency at n_jobs={max_cores}:")
    ms_eff = meanshiftpp_speedup[max_cores]['efficiency']
    gs_eff = gridshift_speedup[max_cores]['efficiency']
    print(f"- MeanShiftPP: {ms_eff:.1%}")
    print(f"- GridShift: {gs_eff:.1%}")
    
    # Optimal n_jobs recommendation
    print(f"\nRecommended n_jobs based on efficiency > 70%:")
    for algo_name, speedup_data in [("MeanShiftPP", meanshiftpp_speedup), ("GridShift", gridshift_speedup)]:
        optimal_njobs = []
        for n_jobs, data in speedup_data.items():
            if data['efficiency'] > 0.7 and n_jobs > 1:
                optimal_njobs.append(n_jobs)
        
        if optimal_njobs:
            recommended = max(optimal_njobs)
            print(f"- {algo_name}: n_jobs={recommended}")
        else:
            print(f"- {algo_name}: n_jobs=1 (parallel overhead too high)")


if __name__ == "__main__":
    main() 