"""
Numba vs C++/Cython MeanShiftPP Benchmark

This benchmark compares the performance and accuracy of:
- Numba-optimized implementation of MeanShift++
- C++/Cython implementation from shiftclustering.MeanShiftPP

The numba implementation follows the exact 3-pass algorithm from the C++ version:
1. First pass: Bin all points and create means map for bins with points
2. Second pass: Accumulate contributions only from neighbor bins that contain points  
3. Third pass: Update each point to the mean of its own bin
"""

import statistics
import time

import numpy as np
import numba
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score

from shiftclustering import MeanShiftPP


@numba.jit(nopython=True, cache=True)
def generate_offsets_numba(d: int, base: int = 3) -> np.ndarray:
    """
    Generate offset array for neighbor computation.
    Numba-optimized for performance.
    
    Parameters
    ----------
    d : int
        Number of dimensions
    base : int
        Base for offsets (3 for (-1, 0, 1))
        
    Returns
    -------
    np.ndarray
        Offset array of shape (base**d, d)
    """
    num_offsets = base ** d
    offsets = np.zeros((num_offsets, d), dtype=np.int32)
    
    for i in range(num_offsets):
        temp = i
        for j in range(d):
            offsets[i, j] = (temp % base) - 1  # Convert to -1, 0, 1
            temp //= base
    
    return offsets


@numba.jit(nopython=True, cache=True)
def meanshiftpp_numba_step(X_shifted: np.ndarray, bandwidth: float, offsets: np.ndarray) -> np.ndarray:
    """
    Single step of MeanShift++ algorithm using numba optimization.
    Follows the exact 3-pass algorithm from the C++ version.
    
    Parameters
    ----------
    X_shifted : np.ndarray
        Current shifted points of shape (n, d)
    bandwidth : float
        Bandwidth for binning
    offsets : np.ndarray
        Offset array of shape (base**d, d)
        
    Returns
    -------
    np.ndarray
        New shifted points after one iteration
    """
    n, d = X_shifted.shape
    base_d = offsets.shape[0]
    
    # FIRST PASS: Bin all points (using floor division like C++)
    bins = np.floor(X_shifted / bandwidth).astype(np.int32)
    
    # Find unique bins that contain points
    # Use hash-based approach since numba doesn't support dictionaries
    # We'll track which bins exist by checking all points
    max_coord = min(np.max(np.abs(bins)) + 10, 1000)  # Safety margin, but prevent overflow
    offset_multiplier = 2 * max_coord + 1
    
    # Create hash values for bins
    bin_hashes = np.zeros(n, dtype=np.int32)
    for i in range(n):
        hash_val = 0
        for j in range(d):
            hash_val = hash_val * offset_multiplier + bins[i, j] + max_coord
        bin_hashes[i] = hash_val
    
    # Find unique bin hashes
    unique_hashes = np.unique(bin_hashes)
    num_unique = len(unique_hashes)
    
    # Handle empty case
    if num_unique == 0:
        return X_shifted.copy()
    
    # Reconstruct bins from hashes
    unique_bins_array = np.zeros((num_unique, d), dtype=np.int32)
    for i, hash_val in enumerate(unique_hashes):
        temp = hash_val
        for j in range(d-1, -1, -1):
            unique_bins_array[i, j] = (temp % offset_multiplier) - max_coord
            temp //= offset_multiplier
    
    # Create mapping from hash to index
    max_hash = np.max(unique_hashes)
    hash_to_idx = np.full(max_hash + 1, -1, dtype=np.int32)
    for i, hash_val in enumerate(unique_hashes):
        hash_to_idx[hash_val] = i
    
    # Initialize sums and counts for bins with points
    bin_sums = np.zeros((num_unique, d), dtype=np.float32)
    bin_counts = np.zeros(num_unique, dtype=np.float32)
    
    # SECOND PASS: Accumulate means (only for neighbor bins that contain points)
    for i in range(n):
        point_bin = bins[i]
        
        # Check all neighbors of this point's bin
        for j in range(base_d):
            # Get neighbor bin coordinates
            neighbor_bin = point_bin + offsets[j]
            
            # Compute hash for neighbor bin
            neighbor_hash = 0
            for k in range(d):
                neighbor_hash = neighbor_hash * offset_multiplier + neighbor_bin[k] + max_coord
            
            # Check if this neighbor bin contains points
            if 0 <= neighbor_hash < len(hash_to_idx) and hash_to_idx[neighbor_hash] >= 0:
                neighbor_idx = hash_to_idx[neighbor_hash]
                bin_sums[neighbor_idx] += X_shifted[i]
                bin_counts[neighbor_idx] += 1.0
    
    # THIRD PASS: Update each point to the mean of its own bin
    X_shifted_new = X_shifted.copy()
    for i in range(n):
        point_hash = bin_hashes[i]
        if 0 <= point_hash < len(hash_to_idx) and hash_to_idx[point_hash] >= 0:
            bin_idx = hash_to_idx[point_hash]
            if bin_counts[bin_idx] > 0:
                X_shifted_new[i] = bin_sums[bin_idx] / bin_counts[bin_idx]
    
    return X_shifted_new


def meanshiftpp_numba(X: np.ndarray, bandwidth: float, n_steps: int, tol: float = 1e-3,
                     base: int = 3) -> np.ndarray:
    """
    Mean shift clustering with numba optimization.
    
    This implementation follows the exact 3-pass algorithm from the C++ version.

    Parameters
    ----------
    X : np.ndarray
        (n, d) array of points.
    bandwidth : float
        Radius for binning points.
    n_steps : int
        Number of iterations.
    tol : float
        Tolerance for convergence.
    base : int
        Base for offsets. Default is 3.

    Returns
    -------
    np.ndarray
        (n, d) array of new points after mean shift clustering.
    """
    X_shifted = X.astype(np.float32).copy()
    n, d = X_shifted.shape

    # Generate offsets once (numba-optimized)
    offsets = generate_offsets_numba(d, base)

    for iteration in range(n_steps):
        # Perform one step of mean shift
        X_shifted_new = meanshiftpp_numba_step(X_shifted, bandwidth, offsets)

        # Check for convergence
        max_norm = np.max(np.linalg.norm(X_shifted_new - X_shifted, axis=1))
        if max_norm <= tol:
            print(f"Numba MeanShift++ converged at {iteration + 1} steps.")
            break

        X_shifted = X_shifted_new

    return X_shifted


def numba_meanshift_clustering(X: np.ndarray, bandwidth: float, n_steps: int, tol: float = 1e-3) -> np.ndarray:
    """
    Complete clustering pipeline using numba MeanShift++.
    
    Returns cluster labels similar to MeanShiftPP.fit_predict().
    """
    X_shifted = meanshiftpp_numba(X, bandwidth, n_steps, tol)
    
    # Find unique cluster centers (similar to MeanShiftPP)
    cluster_centers, labels = np.unique(X_shifted, return_inverse=True, axis=0)
    
    return labels


def benchmark_implementations(
    X_np: np.ndarray, 
    bandwidth: float, 
    max_iter: int = 300, 
    threshold: float = 1e-4,
    n_runs: int = 3
) -> dict[str, dict]:
    """
    Benchmark both implementations with identical parameters.
    
    Parameters
    ----------
    X_np : np.ndarray
        Input data as numpy array
    bandwidth : float
        Bandwidth parameter
    max_iter : int
        Maximum iterations
    threshold : float
        Convergence threshold
    n_runs : int
        Number of runs for statistical stability
        
    Returns
    -------
    Dict with results for both implementations
    """
    
    results = {
        'numba': {'times': [], 'labels': [], 'n_clusters': []},
        'cython': {'times': [], 'labels': [], 'n_clusters': []}
    }
    
    print(f"Benchmarking with {len(X_np)} samples, {X_np.shape[1]} features")
    print(f"Parameters: bandwidth={bandwidth}, max_iter={max_iter}, threshold={threshold}")
    
    # Benchmark Numba implementation
    print("\nTesting Numba implementation...")
    for run in range(n_runs):
        start_time = time.time()
        labels_numba = numba_meanshift_clustering(X_np, bandwidth, max_iter, threshold)
        end_time = time.time()
        
        elapsed = end_time - start_time
        n_clusters = len(np.unique(labels_numba))
        
        results['numba']['times'].append(elapsed)
        results['numba']['labels'].append(labels_numba)
        results['numba']['n_clusters'].append(n_clusters)
        
        print(f"  Run {run+1}: {elapsed:.4f}s, {n_clusters} clusters")
    
    # Benchmark C++/Cython implementation
    print("\nTesting C++/Cython implementation...")
    for run in range(n_runs):
        # Create fresh instance for each run
        ms = MeanShiftPP(bandwidth=bandwidth, threshold=threshold, max_iter=max_iter)
        
        start_time = time.time()
        labels_cython = ms.fit_predict(X_np)
        end_time = time.time()
        
        elapsed = end_time - start_time
        n_clusters = len(np.unique(labels_cython))
        
        results['cython']['times'].append(elapsed)
        results['cython']['labels'].append(labels_cython)
        results['cython']['n_clusters'].append(n_clusters)
        
        print(f"  Run {run+1}: {elapsed:.4f}s, {n_clusters} clusters")
    
    return results


def calculate_metrics(results: dict[str, dict], X: np.ndarray, y_true: np.ndarray | None = None) -> dict[str, dict]:
    """Calculate performance and quality metrics."""
    metrics = {}
    
    for impl_name, data in results.items():
        times = data['times']
        labels_list = data['labels']
        n_clusters_list = data['n_clusters']
        
        # Use first run for quality metrics
        labels = labels_list[0]
        
        # Performance metrics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        mean_clusters = statistics.mean(n_clusters_list)
        
        # Quality metrics
        quality_metrics = {}
        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
            try:
                quality_metrics['silhouette'] = silhouette_score(X, labels)
            except ValueError as e:
                print(f"Silhouette score not available: {e}")
                quality_metrics['silhouette'] = -1.0
        else:
            quality_metrics['silhouette'] = -1.0
            
        if y_true is not None:
            quality_metrics['ari'] = adjusted_rand_score(y_true, labels)
        
        metrics[impl_name] = {
            'mean_time': mean_time,
            'std_time': std_time,
            'mean_clusters': mean_clusters,
            'all_times': times,
            **quality_metrics
        }
    
    return metrics


def compare_results(results: dict[str, dict]) -> dict[str, float]:
    """Compare the similarity of clustering results between implementations."""
    numba_labels = results['numba']['labels'][0]
    cython_labels = results['cython']['labels'][0]
    
    # Calculate Adjusted Rand Index between the two implementations
    ari = adjusted_rand_score(numba_labels, cython_labels)
    
    return {'ari_between_implementations': ari}


def print_results(metrics: dict[str, dict], comparison: dict[str, float]) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'='*90}")
    print("NUMBA vs C++/CYTHON MEANSHIFT++ BENCHMARK RESULTS")
    print(f"{'='*90}")
    
    print(f"{'Implementation':<15} {'Time (s)':<12} {'±Std':<8} {'Clusters':<10} {'Silhouette':<12} {'ARI':<8}")
    print("-" * 90)
    
    for impl_name, data in metrics.items():
        display_name = "Numba" if impl_name == "numba" else "C++/Cython"
        ari_str = f"{data.get('ari', 0.0):.4f}" if 'ari' in data else "N/A"
        
        print(f"{display_name:<15} "
              f"{data['mean_time']:<12.4f} "
              f"±{data['std_time']:<7.4f} "
              f"{data['mean_clusters']:<10.1f} "
              f"{data['silhouette']:<12.4f} "
              f"{ari_str:<8}")
    
    # Speedup analysis
    numba_time = metrics['numba']['mean_time']
    cython_time = metrics['cython']['mean_time']
    
    if numba_time < cython_time:
        speedup = cython_time / numba_time
        faster = "Numba"
    else:
        speedup = numba_time / cython_time
        faster = "C++/Cython"
    
    print(f"\nPerformance Analysis:")
    print(f"- {faster} is {speedup:.2f}x faster")
    print(f"- Implementation Agreement (ARI): {comparison['ari_between_implementations']:.4f}")
    
    if comparison['ari_between_implementations'] < 0.8:
        print("  ⚠️  Low agreement between implementations - results may differ significantly")
    elif comparison['ari_between_implementations'] > 0.95:
        print("  ✅ High agreement between implementations - results are very similar")
    else:
        print("  ⚠️  Moderate agreement between implementations")


def main():
    """Run the benchmark with different dataset configurations."""
    
    # Test configurations
    configs = [
        {"n_samples": 1000, "n_features": 2, "n_centers": 5, "bandwidth": 1.0},
        {"n_samples": 3000, "n_features": 3, "n_centers": 10, "bandwidth": 1.5},
        {"n_samples": 5000, "n_features": 4, "n_centers": 15, "bandwidth": 2.0},
    ]
    
    max_iter = 300
    threshold = 1e-4
    n_runs = 3
    random_state = 42
    
    print("Numba vs C++/Cython MeanShift++ Benchmark")
    print(f"Parameters: max_iter={max_iter}, threshold={threshold}, runs={n_runs}")
    print(f"Random state: {random_state}")
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*100}")
        print(f"TEST {i}/{len(configs)}: {config['n_samples']} samples, {config['n_features']}D, {config['n_centers']} centers")
        print(f"{'='*100}")
        
        # Generate dataset
        X, y_true = make_blobs(
            n_samples=config['n_samples'],
            n_features=config['n_features'], 
            centers=config['n_centers'],
            random_state=random_state,
            cluster_std=1.0
        )
        
        print(f"Dataset shape: {X.shape}")
        print(f"True clusters: {len(np.unique(y_true))}")
        
        # Run benchmark
        results = benchmark_implementations(
            X, config['bandwidth'], max_iter, threshold, n_runs
        )
        
        # Calculate metrics
        metrics = calculate_metrics(results, X, y_true)
        comparison = compare_results(results)
        
        # Print results
        print_results(metrics, comparison)
    
    print(f"\n{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print("Key Findings:")
    print("- Both implementations should produce similar clustering results")
    print("- C++/Cython typically faster due to optimized memory management")
    print("- Numba provides good performance with Python flexibility")
    print("- Algorithm equivalence: same 3-pass structure as C++ version")


if __name__ == "__main__":
    main()
